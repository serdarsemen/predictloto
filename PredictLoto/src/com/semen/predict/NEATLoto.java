/*
 */
package com.semen.predict;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.log4j.Logger;
import org.encog.Encog;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;

import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATTraining;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLRegression;
import org.encog.neural.networks.training.TrainingSetScore;

import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.obj.SerializeObject;
import org.encog.util.simple.EncogUtility;
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

	public static double MINVALUE = 0.4; // 0.0009;
	public static double IDEALVALUE = 1.0;
	public static SortedMap<Integer, Double> PREDICTMAPRESULT = new TreeMap<Integer, Double>();
	public static SortedMap<Integer, Double> PREDICTMAP = new TreeMap<Integer, Double>();
	public static SortedMap<Integer, Double> PREDICTLOWMAPRESULT = new TreeMap<Integer, Double>();
	public static SortedMap<Integer, Double> PREDICTLOWMAP = new TreeMap<Integer, Double>();

	// For each file, you'll need a separate Logger.
	// private static Logger log = * Logger.getLogger( JordanLoto.class )
	// private static Logger connectionsLog = Logger.getLogger( "connections." +
	// JordanLoto.class.getName() )
	// private static Logger stacktracesLog = Logger.getLogger( "stacktraces." +
	// JordanLoto.class.getName() )
	// private static Logger httpLog = Logger.getLogger( "http." +
	// JordanLoto.class.getName() )

	
	/**
	 * Evaluate the network and display (to the logger) the output for every
	 * value in the training set. Displays ideal and actual.
	 * 
	 * @param network
	 *            The network to evaluate.
	 * @param training
	 *            The training set to evaluate.
	 */
	public static void evaluate(final MLRegression network,
			final MLDataSet training) {
		for (final MLDataPair pair : training) {
			final MLData output = network.compute(pair.getInput());
			log.debug("Input= "
					+ EncogUtility.formatNeuralData(pair.getInput()));
			log.debug("Actual=" + EncogUtility.formatNeuralData(output));
			log.debug("Ideal= "
					+ EncogUtility.formatNeuralData(pair.getIdeal()));
			log.debug("Success Report ---------");
			calculateSuccess(output, pair.getIdeal());
		}
	}

	/**
	 * Calculate success
	 * 
	 * @param actual
	 *            
	 * @param actual
	 *            ideal
	 * @return The success count
	 */
	public static void calculateSuccess(final MLData actual, final MLData ideal) {

		int counterSuccess = 0;
		int counterLowSuccess = 0;
		int counterTotalPredict = 0;
		int counterLowTotalPredict = 0;
		PREDICTMAP.clear();
		PREDICTMAPRESULT.clear();
		PREDICTLOWMAP.clear();
		PREDICTLOWMAPRESULT.clear();
		for (int i = 0; i < actual.size(); i++) {
			if (ideal.getData(i) == IDEALVALUE) // 1.0
				if (actual.getData(i) > MINVALUE) {
					counterSuccess++;
					PREDICTMAPRESULT.put(i + 1, actual.getData(i));
				} else {
					counterLowSuccess++;
					PREDICTLOWMAPRESULT.put(i + 1, actual.getData(i));
				}
		}
		for (int i = 0; i < actual.size(); i++) {
			if (actual.getData(i) > MINVALUE) {
				counterTotalPredict++;
				PREDICTMAP.put(i + 1, actual.getData(i));
			} else {
				counterLowTotalPredict++;
				PREDICTLOWMAP.put(i + 1, actual.getData(i));
			}
		}
		log.debug("Successfull Predict Count= " + counterSuccess);
		log.debug("Result= " + PREDICTMAPRESULT);
		log.debug("Prediction= " + PREDICTMAP);
		log.debug("Total Predict Count= " + counterTotalPredict);

		log.debug("Successfull Low Predict Count<  " + MINVALUE + " = "
				+ counterLowSuccess);
		log.debug("Low Result= " + PREDICTLOWMAPRESULT);
		log.debug("Low Prediction= " + PREDICTLOWMAP);
		log.debug("Total Low Predict Count= " + counterLowTotalPredict);

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
	public static void trainToError(final MLTrain train, final double error) {

		int epoch = 1;

		log.debug("Beginning training...");
		do {
			train.iteration();

			log.debug("Iteration #" + Format.formatInteger(epoch) + " Error:"
					+ Format.formatPercent(train.getError())
					+ " Target Error: " + Format.formatPercent(error));
			epoch++;
		} while ((train.getError() > error) && !train.isTrainingDone());
		train.finishTraining();
	}

	public NEATNetwork trainAndSave(int sourceTrainData) {

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

		NEATPopulation pop = new NEATPopulation(ConfigLoto.INPUT_SIZE,
				ConfigLoto.IDEAL_SIZE, ConfigLoto.NEATPOPULATIONSIZE);

		// not required, but speeds training if added startes from 40 instead of
		// 31
		pop.setInitialConnectionDensity(ConfigLoto.NEATPOPULATIONDENSITY);
		pop.reset();

		CalculateScore score = new TrainingSetScore(trainingSet);

		// train the neural network
		final NEATTraining train = new NEATTraining(score, pop);
		log.debug("Training NEAT network");
		trainToError(train, ConfigLoto.NEATDESIREDERROR);

		NEATNetwork network = (NEATNetwork) train.getMethod();

		try {
			// for neat save is used
		//	SerializeObject.save(new File(ConfigLoto.NEAT_FILENAME), network);
			// Save NEAT Network
			// only pop
			 EncogDirectoryPersistence.saveObject( new	 File(ConfigLoto.NEAT_FILENAME), pop);
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
				network = (NEATNetwork) SerializeObject.load(new File(
						ConfigLoto.NEAT_FILENAME));
			} catch (ClassNotFoundException | IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

		final MLDataSet testSet = new SQLNeuralDataSet(ConfigLoto.TESTSQL,
				ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
				ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL, ConfigLoto.SQL_UID,
				ConfigLoto.SQL_PWD);

		double e = network.calculateError(testSet);
		log.debug("Loaded network's error is: " + e);

		// test the neural network
		log.debug("****     Neural Network Results:");
		evaluate(network, testSet);

	}

	public static void main(String[] args) {
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

			NEATNetwork neatNetwork = null;
			File networkFile = null;
			

			
			if (arg1 != null) {
				// use the previous saved eg file so no training
				try {
					networkFile = new File(ConfigLoto.NEAT_FILENAME);
					// neatNetwork = (NEATNetwork)
					// EncogDirectoryPersistence.loadObject(new
					// File(ConfigLoto.NEAT_FILENAME));

					
					if (!networkFile.exists()) {
						log.debug("Can't read Neat eg file: " + networkFile.getAbsolutePath());
						neatNetwork = program
								.trainAndSave(ConfigLoto.DATASOURCESQL);

					} else {
						neatNetwork = (NEATNetwork) EncogDirectoryPersistence
								.loadObject(networkFile);
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
			Encog.getInstance().shutdown();
		}

	}
}
