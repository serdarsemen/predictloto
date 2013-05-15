/*
 */
package com.semen.predict;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.encog.Encog;

import org.encog.ml.data.MLDataSet;

import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.hyperneat.substrate.SubstrateFactory;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.neat.training.species.OriginalNEATSpeciation;
import org.encog.ml.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;

import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.obj.SerializeObject;
import org.encog.util.simple.TrainingSetUtil;
import com.semen.util.MySQLUtil;

/**
 * NEATLoto: This network solves Loto neural network problem. It uses a
 * NEAT evolving network.
 * 
 * @author serdar semen
 * @version 1.0
 */
public class NEATLoto {
	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(NEATLoto.class); // .getName());
	private static final long serialVersionUID = -3656587890L;
	private static Properties prop = new Properties();

	
	public static void createInsertSQLPart1(final double desired_Error){
		if (ConfigLoto.ISHYPERNEAT == ConfigLoto.NEATMODE) {
			ConfigLoto.INSERTSAYISALPREDICTPART1 = "\"NEAT\"," + desired_Error + ","
					+ ConfigLoto.NEATPOPULATIONSIZE + ","
					+ ConfigLoto.NEATPOPULATIONDENSITY + ",";
		} else if (ConfigLoto.ISHYPERNEAT == ConfigLoto.HYPERNEATMODE) {
			ConfigLoto.INSERTSAYISALPREDICTPART1 = "\"HYPERNEAT\"," + desired_Error
					+ "," + ConfigLoto.NEATPOPULATIONSIZE + ","
					+ ConfigLoto.NEATPOPULATIONDENSITY + ",";
		} else {
			ConfigLoto.INSERTSAYISALPREDICTPART1 = "\"NEAT\"," + desired_Error + ","
					+ ConfigLoto.NEATPOPULATIONSIZE + ","
					+ ConfigLoto.NEATPOPULATIONDENSITY + ",";
		}
	}
	
	
	/**
	 * Train to a specific error, using the specified training method, send the
	 * output to the logger.
	 * 
	 * @param train
	 *            The training method.
	 * @param error
	 *            The desired error level.
	 */
	public static void trainNetwork(final EvolutionaryAlgorithm train,
			final double desired_Error) {

		int epoch = 1;
		double trainError = 1.0;
		double prevtrainError = 1.0;
		double sameErrorCount = 0;

		String strTargetError = Format.formatDouble(desired_Error, 4);
		log.debug("ISHYPERNEAT= " + ConfigLoto.ISHYPERNEAT);
		log.debug("LO_WEEKNO= " + ConfigLoto.LO_WEEKNO);
		log.debug("HI_WEEKNO= " + ConfigLoto.HI_WEEKNO);

		log.debug("NEATPOPULATIONSIZE= " + ConfigLoto.NEATPOPULATIONSIZE);
		log.debug("NEATPOPULATIONDENSITY= " + ConfigLoto.NEATPOPULATIONDENSITY);

		log.debug("Beginning NEAT training...");
		do {
			prevtrainError = trainError;
			train.iteration();
			trainError = train.getError();
			log.debug("NEAT # " + Format.formatInteger(epoch) + " Err= "
					+ Format.formatDouble(trainError, 4) + " Target Err= "
					+ strTargetError + ", Species= "
					+ train.getPopulation().getSpecies().size());

			// Save error
			if ((epoch % ConfigLoto.EPOCHSAVEINTERVAL) == 0) {
				log.debug("Saving NEAT POP / network  Epoch #" + epoch);

				// Save NEAT pop
				EncogDirectoryPersistence.saveObject(new File(
						ConfigLoto.NEAT_FILENAME), train.getPopulation());

				NEATNetwork network = (NEATNetwork) train.getCODEC().decode(
						train.getBestGenome());
				try {
					// Save NEAT network
					SerializeObject.save(new File(
							ConfigLoto.NEAT_SERIALFILENAME), network);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			epoch++;
			if (prevtrainError == trainError) {
				sameErrorCount++;
				// log.debug("SameErrorCount=" + sameErrorCount + "preverr "
				// + prevtrainError + "trainerr " + trainError);
			} else {
				// log.debug("SameErrorCount=0"+"preverr "+
				// prevtrainError+"trainerr "+ trainError);
				sameErrorCount = 0;
			}
		} while ((train.getError() > desired_Error)
				&& (sameErrorCount < ConfigLoto.NEATEPOCHEXITCOUNTER));
		train.finishTraining();

		createInsertSQLPart1(desired_Error);
	}

	/*
	 * Continue training from the last saved network
	 */
	public NEATNetwork loadAndContinueTrain(final int sourceTrainData,
			NEATNetwork network, NEATPopulation pop) {
		if (network == null) {
			log.debug("Loading NEAT network");

			try {
				network = (NEATNetwork) SerializeObject.load(new File(
						ConfigLoto.NEAT_SERIALFILENAME));
			} catch (ClassNotFoundException | IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

		if (pop == null) {
			log.debug("Loading NEAT pop");
			pop = (NEATPopulation) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.NEAT_FILENAME));
		}

		MLDataSet trainingSet = null;
		if (sourceTrainData == ConfigLoto.DATASOURCESQL)
			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL, MySQLUtil.SQL_UID,
					MySQLUtil.SQL_PWD);
		else if (sourceTrainData == ConfigLoto.DATASOURCECSV)
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);
		else
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);

		if (network != null) {
			double e = network.calculateError(trainingSet);
			log.debug("Loaded NEAT network's error for previous train set is: "
					+ e);
			createInsertSQLPart1(e);
		} else {
			log.debug("NEAT network is NULL");
		}

		CalculateScore score = new TrainingSetScore(trainingSet);
		BoxesScore boxScore = null;

		// train the neural network

		EvolutionaryAlgorithm train = null;

		if (ConfigLoto.ISHYPERNEAT == ConfigLoto.NEATMODE) {
			train = NEATUtil.constructNEATTrainer(pop, score);

		} else if (ConfigLoto.ISHYPERNEAT == ConfigLoto.HYPERNEATMODE) {
			Substrate substrate = SubstrateFactory.factorSandwichSubstrate(
					ConfigLoto.BASE_RESOLUTION, ConfigLoto.BASE_RESOLUTION);
			boxScore = new BoxesScore(ConfigLoto.BASE_RESOLUTION);
			if (pop == null) {
				pop = new NEATPopulation(substrate,
						ConfigLoto.NEATPOPULATIONSIZE);
				pop.setActivationCycles(4);
				pop.reset();
			}
			train = NEATUtil.constructNEATTrainer(pop, boxScore);
			OriginalNEATSpeciation speciation = new OriginalNEATSpeciation();
			speciation.setCompatibilityThreshold(1);
			train.setSpeciation(speciation);
		}

		NEATLoto.trainNetwork(train, ConfigLoto.NEATDESIREDERROR);
		network = (NEATNetwork) train.getCODEC().decode(train.getBestGenome());

		try {
			// Save pop
			EncogDirectoryPersistence.saveObject(new File(
					ConfigLoto.NEAT_FILENAME), pop);
			// Save NEAT Network
			SerializeObject.save(new File(ConfigLoto.NEAT_SERIALFILENAME),
					network);
		} catch (Throwable t) {
			t.printStackTrace();
		}
		return network;
	}

	public NEATNetwork trainAndSave(int sourceTrainData) {

		EvolutionaryAlgorithm train = null;
		NEATPopulation pop = null;
		MLDataSet trainingSet = null;

		if (sourceTrainData == ConfigLoto.DATASOURCESQL)
			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL, MySQLUtil.SQL_UID,
					MySQLUtil.SQL_PWD);

		else if (sourceTrainData == ConfigLoto.DATASOURCECSV)
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);
		else
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);

		CalculateScore score = new TrainingSetScore(trainingSet);
		BoxesScore boxScore = null;

		if (ConfigLoto.ISHYPERNEAT == ConfigLoto.NEATMODE) {

			pop = new NEATPopulation(ConfigLoto.INPUT_SIZE,
					ConfigLoto.IDEAL_SIZE, ConfigLoto.NEATPOPULATIONSIZE);

			// not required, but speeds training if added starts from 40 instead
			// of
			// 31
			pop.setInitialConnectionDensity(ConfigLoto.NEATPOPULATIONDENSITY);
			pop.reset();

			// train the neural network
			train = NEATUtil.constructNEATTrainer(pop, score);
		} else if (ConfigLoto.ISHYPERNEAT == ConfigLoto.HYPERNEATMODE)

		{
			Substrate substrate = SubstrateFactory
					.factorSandwichSubstrate(7, 7);
			boxScore = new BoxesScore(7);
			pop = new NEATPopulation(substrate, ConfigLoto.NEATPOPULATIONSIZE);
			pop.setActivationCycles(4);
			pop.reset();
			train = NEATUtil.constructNEATTrainer(pop, boxScore);
			OriginalNEATSpeciation speciation = new OriginalNEATSpeciation();
			speciation.setCompatibilityThreshold(1);
			train.setSpeciation(speciation);
		}

		NEATLoto.trainNetwork(train, ConfigLoto.NEATDESIREDERROR);
		NEATNetwork network = (NEATNetwork) train.getCODEC().decode(
				train.getBestGenome());
		try {
			// Save pop
			EncogDirectoryPersistence.saveObject(new File(
					ConfigLoto.NEAT_FILENAME), pop);

			// train.dump(new File(ConfigLoto.NEAT_DUMPFILENAME));

			// Save NEAT Network
			SerializeObject.save(new File(ConfigLoto.NEAT_SERIALFILENAME),
					network);
		} catch (Throwable t) {
			t.printStackTrace();
		}
		return network;
	}

	public void loadAndEvaluate(NEATNetwork network) {

		if (network == null) {
			log.debug("Loading NEAT network");
			/*
			 * network = (NEATNetwork) EncogDirectoryPersistence .loadObject(new
			 * File(ConfigLoto.NEAT_FILENAME));
			 */
			try {
				// network = (NEATNetwork) SerializeObject.load(new File(
				// ConfigLoto.NEAT_FILENAME));
				network = (NEATNetwork) SerializeObject.load(new File(
						ConfigLoto.NEAT_SERIALFILENAME));
			} catch (ClassNotFoundException | IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

		final MLDataSet testSet = new SQLNeuralDataSet(ConfigLoto.TESTSQL,
				ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
				MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL, MySQLUtil.SQL_UID,
				MySQLUtil.SQL_PWD);
		if (testSet.size() > 0) {
			double e = network.calculateError(testSet);
			log.debug("Loaded network's error is: " + e);

			createInsertSQLPart1(e);
			// test the neural network
			log.debug("****     Neural Network Results:");
			ConfigLoto.evaluate(network, testSet);
		} else {
			log.debug("Test set is empty");
		}
	}

	public static void main(String[] args) {
		long startTime = System.nanoTime();
		try {

			String arg1 = null;
			if (args.length != 0) {
				arg1 = args[0]; // means load eg file
			}

			// load a properties file from class path, inside static method
			prop.load(NEATLoto.class.getClassLoader().getResourceAsStream(
					"config.properties"));

			// get the property value and print it out
			/*
			 * System.out.println(prop.getProperty("database"));
			 */

			NEATLoto program = new NEATLoto();

			NEATPopulation pop = null;
			NEATNetwork neatNetwork = null;
			File networkFile = null;
			// File networkSerFile = null;
			if (arg1 != null) {
				// use the previous saved eg file so no training
				try {
					networkFile = new File(ConfigLoto.NEAT_FILENAME);
					// networkSerFile = new
					// File(ConfigLoto.NEAT_SERIALFILENAME);
					// neatNetwork = (NEATNetwork)
					// EncogDirectoryPersistence.loadObject(new
					// File(ConfigLoto.NEAT_FILENAME));

					if (!networkFile.exists()) {
						log.debug("Can't read Neat eg file: "
								+ networkFile.getAbsolutePath());
						neatNetwork = program
								.trainAndSave(ConfigLoto.DATASOURCESQL);

					} else {/*
							 * pop= (NEATPopulation)
							 * EncogDirectoryPersistence.loadObject
							 * (networkFile); neatNetwork = (NEATNetwork)
							 * SerializeObject .load(networkSerFile);
							 */
						neatNetwork = program.loadAndContinueTrain(
								ConfigLoto.DATASOURCESQL, neatNetwork, pop);
					}
				} catch (Throwable t) {
					t.printStackTrace();
					neatNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);

				} finally {
				}
				if (neatNetwork == null) {
					neatNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);
				}
				program.loadAndEvaluate(neatNetwork);
			} else {
				neatNetwork = program.trainAndSave(ConfigLoto.DATASOURCESQL);
				program.loadAndEvaluate(neatNetwork);
			}
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			double estimatedTimeMin = (System.nanoTime() - startTime) / 60000000000.0;
			log.debug("Elapsed Time = " + ConfigLoto.round2(estimatedTimeMin)
					+ " (min) ");
			Encog.getInstance().shutdown();
		}
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}
}
