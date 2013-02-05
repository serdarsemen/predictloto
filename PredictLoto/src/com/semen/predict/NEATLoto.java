/*
 */
package com.semen.predict;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.encog.Encog;

import org.encog.ml.data.MLDataSet;

import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATTraining;
import org.encog.ml.CalculateScore;
import org.encog.neural.networks.BasicNetwork;
//import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;

import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
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

	// For each file, you'll need a separate Logger.
	// private static Logger log = * Logger.getLogger( JordanLoto.class )
	// private static Logger connectionsLog = Logger.getLogger( "connections." +
	// JordanLoto.class.getName() )
	// private static Logger stacktracesLog = Logger.getLogger( "stacktraces." +
	// JordanLoto.class.getName() )
	// private static Logger httpLog = Logger.getLogger( "http." +
	// JordanLoto.class.getName() )

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
		EncogUtility.trainToError(train, ConfigLoto.NEATDESIREDERROR);

		NEATNetwork network = (NEATNetwork) train.getMethod();

		try {
			// for neat save is used
			SerializeObject.save(new File(ConfigLoto.NEAT_FILENAME), network);
			// Save NEAT Network
			// only pop
			// EncogDirectoryPersistence.saveObject( new
			// File(ConfigLoto.NEAT_FILENAME), network);
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
		EncogUtility.evaluate(network, testSet);

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
			
			final File networkFile = new File(ConfigLoto.NEAT_FILENAME);

			if (!networkFile.exists()) {
				log.debug("Can't read Neat eg file: " + networkFile.getAbsolutePath());
			} else {

				neatNetwork = (NEATNetwork) EncogDirectoryPersistence
						.loadObject(networkFile);
			}

			if (arg1 != null) {
				// use the previous saved eg file so no training
				try {
					// neatNetwork = (NEATNetwork)
					// EncogDirectoryPersistence.loadObject(new
					// File(ConfigLoto.NEAT_FILENAME));

						neatNetwork = (NEATNetwork) SerializeObject
								.load(new File(ConfigLoto.NEAT_FILENAME));

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
