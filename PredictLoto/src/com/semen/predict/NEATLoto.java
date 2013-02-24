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
import org.encog.neural.neat.training.NEATTraining;
import org.encog.neural.neat.training.species.OriginalNEATSpeciation;
import org.encog.ml.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;

import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.obj.SerializeObject;
import org.encog.util.simple.TrainingSetUtil;

/**
 * NEATLoto: This network solves Loto neural network problem. However, it uses a
 * NEAT evolving network.
 * 
 * @author serdar semen
 * @version 1.0
 */
public class NEATLoto {
	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(NEATLoto.class); // .getName());
	// private static final long serialVersionUID = 3L;
	private static Properties prop = new Properties();

	// For each file, you'll need a separate Logger.
	// private static Logger log = * Logger.getLogger( JordanLoto.class )
	// private static Logger connectionsLog = Logger.getLogger( "connections." +
	// JordanLoto.class.getName() )
	// private static Logger stacktracesLog = Logger.getLogger( "stacktraces." +
	// JordanLoto.class.getName() )
	// private static Logger httpLog = Logger.getLogger( "http." +
	// JordanLoto.class.getName() )

	/**
	 * Train to a specific error, using the specified training method, send the
	 * output to the logger.
	 * 
	 * @param train
	 *            The training method.
	 * @param error
	 *            The desired error level.
	 */
	public static void trainToError(final NEATTraining train,
			final double error, NEATPopulation pop) {

		int epoch = 1;
		double train_Error = 1.0;
		String str_TargetError = Format.formatDouble(error, 4);
		log.debug("ISHYPERNEAT= " +ConfigLoto.ISHYPERNEAT);
		log.debug("LO_WEEKNO= " +ConfigLoto.LO_WEEKNO);
		log.debug("HI_WEEKNO= " +ConfigLoto.HI_WEEKNO);

		log.debug("NEATPOPULATIONSIZE= " +ConfigLoto.NEATPOPULATIONSIZE);
		log.debug("NEATPOPULATIONDENSITY= " +ConfigLoto.NEATPOPULATIONDENSITY);

		log.debug("Beginning NEAT training...");
		do {
			train.iteration();
			train_Error = train.getError();
			log.debug("NEAT # " + Format.formatInteger(epoch) + " Err= "
					+ Format.formatDouble(train_Error, 4) + " Target Err= "
					+ str_TargetError + ", Species= "
					+ train.getNEATPopulation().getSpecies().size());
			if ((epoch % ConfigLoto.EPOCHSAVEINTERVAL) == 0) {
				log.debug("Saving NEAT POP / network  Epoch #" + epoch);

				// Save NEAT pop
				EncogDirectoryPersistence.saveObject(new File(
						ConfigLoto.NEAT_FILENAME), pop);

				NEATNetwork network = (NEATNetwork) train.getMethod();
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
		} while ((train.getError() > error) && !train.isTrainingDone());
		train.finishTraining();
	}

	/*
	 * Continue training from the last saved network
	 */
	public NEATNetwork loadAndContinueTrain(int sourceTrainData,
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
		if (sourceTrainData == 0)
			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL,
					ConfigLoto.SQL_UID, ConfigLoto.SQL_PWD);
		else if (sourceTrainData == 1)
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
		} else {
			log.debug("NEAT network is NULL");
		}
		
		CalculateScore score = new TrainingSetScore(trainingSet);
		BoxesScore boxScore  = null;
		
		// train the neural network

		NEATTraining train = null;

		if (ConfigLoto.ISHYPERNEAT == 0) {
			new NEATTraining(score, pop);
			
		} else {
			Substrate substrate = SubstrateFactory
					.factorSandwichSubstrate(ConfigLoto.BASE_RESOLUTION,ConfigLoto.BASE_RESOLUTION);
			boxScore = new BoxesScore(ConfigLoto.BASE_RESOLUTION);
			if (pop == null) {
				pop = new NEATPopulation(substrate,
						ConfigLoto.NEATPOPULATIONSIZE);
				pop.setActivationCycles(4);
				pop.reset();
			}
			train = new NEATTraining(boxScore, pop);
			OriginalNEATSpeciation speciation = new OriginalNEATSpeciation();
			speciation.setCompatibilityThreshold(1);
			train.setSpeciation(speciation);
		}

		NEATLoto.trainToError(train, ConfigLoto.NEATDESIREDERROR, pop);
		network = (NEATNetwork) train.getMethod();

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

		NEATTraining train = null;
		NEATPopulation pop = null;
		MLDataSet trainingSet = null;

		if (sourceTrainData == 0)
			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL,
					ConfigLoto.SQL_UID, ConfigLoto.SQL_PWD);

		else if (sourceTrainData == 1)
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);
		else
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);

		CalculateScore score = new TrainingSetScore(trainingSet);
		BoxesScore boxScore  = null;
		if (ConfigLoto.ISHYPERNEAT == 0) {

			pop = new NEATPopulation(ConfigLoto.INPUT_SIZE,
					ConfigLoto.IDEAL_SIZE, ConfigLoto.NEATPOPULATIONSIZE);

			// not required, but speeds training if added starts from 40 instead
			// of
			// 31
			pop.setInitialConnectionDensity(ConfigLoto.NEATPOPULATIONDENSITY);
			pop.reset();

			// train the neural network
			train = new NEATTraining(score, pop);
		} else {
			Substrate substrate = SubstrateFactory
					.factorSandwichSubstrate(7, 7);
		    boxScore = new BoxesScore(7);
			pop = new NEATPopulation(substrate, ConfigLoto.NEATPOPULATIONSIZE);
			pop.setActivationCycles(4);
			pop.reset();
			train = new NEATTraining(boxScore, pop);
			OriginalNEATSpeciation speciation = new OriginalNEATSpeciation();
			speciation.setCompatibilityThreshold(1);
			train.setSpeciation(speciation);
		}

		NEATLoto.trainToError(train, ConfigLoto.NEATDESIREDERROR, pop);
		NEATNetwork network = (NEATNetwork) train.getMethod();
		try {
			// Save pop
			EncogDirectoryPersistence.saveObject(new File(
					ConfigLoto.NEAT_FILENAME), pop);
			
			
			train.dump(new File(ConfigLoto.NEAT_DUMPFILENAME));
			
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
				ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL, ConfigLoto.SQL_UID,
				ConfigLoto.SQL_PWD);
		if (testSet.size() > 0) {
			double e = network.calculateError(testSet);
			log.debug("Loaded network's error is: " + e);

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
			// ... the code being measured ...

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
			log.debug("Elapsed Time = " + ConfigLoto.round2(estimatedTimeMin)+ " (min) ");
			Encog.getInstance().shutdown();
		}
	}
}
