/*
 * Predict Loto 1.0
 * Using Encog(tm) Java Examples v3.2
 * @author serdar semen
 * @version 1.0
 */
package com.semen.predict;

import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;

import org.encog.ml.data.MLDataSet;

import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.ml.CalculateScore;

import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.JordanPattern;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

import java.io.File;
import java.util.Properties;

/**
 * Implement an Jordan style neural network with Encog. This network attempts to
 * predict the next value in Loto Jordan is better suited to a larger array of
 * output neurons.
 * 
 * @author serdar semen
 * 
 */
public class JordanLoto {
	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(JordanLoto.class); // .getName());
	private static final long serialVersionUID = -4738357581L;
	private static Properties prop = new Properties();

	/*
	 * For each file, you'll need a separate Logger. private static Logger log =
	 * Logger.getLogger( JordanLoto.class ) private static Logger connectionsLog
	 * = Logger.getLogger( "connections." + JordanLoto.class.getName() ) private
	 * static Logger stacktracesLog = Logger.getLogger( "stacktraces." +
	 * JordanLoto.class.getName() ) private static Logger httpLog =
	 * Logger.getLogger( "http." + JordanLoto.class.getName() )
	 */

	static BasicNetwork createJordanNetwork() {
		// construct an Jordan type network
		log.debug("construct an Jordan type network");
		JordanPattern pattern = new JordanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(ConfigLoto.INPUT_SIZE);
		pattern.addHiddenLayer(ConfigLoto.JORDANHIDDENNEURONSIZE);
		pattern.setOutputNeurons(ConfigLoto.IDEAL_SIZE);
		return (BasicNetwork) pattern.generate();
	}

	static BasicNetwork createFeedforwardNetwork() {
		// construct a feedforward type network
		log.debug("construct a feedforward type network");
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(ConfigLoto.INPUT_SIZE);
		pattern.addHiddenLayer(ConfigLoto.FEEDFORWARDHIDDENNEURONSIZE);
		pattern.setOutputNeurons(ConfigLoto.IDEAL_SIZE);
		return (BasicNetwork) pattern.generate();
	}

	public static double trainNetwork(final String what,
			final BasicNetwork network, final MLDataSet trainingSet) {
		// train the neural network
		log.debug("Train " + what + " Network");
		CalculateScore score = new TrainingSetScore(trainingSet);
		final MLTrain trainAlt = new NeuralSimulatedAnnealing(network, score,
				ConfigLoto.SIMANNEAL_STARTTEMP, ConfigLoto.SIMANNEAL_STOPTEMP,
				ConfigLoto.SIMANNEAL_CYCLES);

		final MLTrain trainMain = new Backpropagation(network, trainingSet,
				ConfigLoto.BPROP_LEARNRATE, ConfigLoto.BPROP_MOMENTUM);

		final StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		// If below lined is used strategy can not be used..
		// EncogUtility.trainToError(trainMain, ConfigLoto.JORDANDESIREDERROR);

		int epoch = 1;
		double train_Error = 1.0;
		double desired_Error = ConfigLoto.JORDANDESIREDERROR;
		String str_TargetError = Format.formatDouble(desired_Error, 4);
		while ((!stop.shouldStop()) && (train_Error > desired_Error)) {
			trainMain.iteration();
			train_Error = trainMain.getError();
			log.debug("Training " + what + ", Epoch #" + epoch + " Error= "
					+ Format.formatDouble(train_Error, 4) + " Target Error= "
					+ str_TargetError);
			if ((epoch % ConfigLoto.EPOCHSAVEINTERVAL) == 0) {
				log.debug("Saving " + what + ", Epoch #" + epoch);
				// Save feedforward Network
				if (what.equals("Jordan"))
					EncogDirectoryPersistence.saveObject(new File(
							ConfigLoto.JORDAN_FILENAME), network);
				else
					// Save Elman Network
					EncogDirectoryPersistence.saveObject(new File(
							ConfigLoto.ELMANFEEDFORWARD_FILENAME), network);
			}
			epoch++;
		}
		trainMain.finishTraining();

		return train_Error;
	}

	public BasicNetwork trainAndSave(int sourceTrainData) {

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

		final BasicNetwork jordanNetwork = JordanLoto.createJordanNetwork();

		final double jordanError = JordanLoto.trainNetwork("Jordan",
				jordanNetwork, trainingSet);

		// Save Jordan Network
		EncogDirectoryPersistence.saveObject(new File(
				ConfigLoto.JORDAN_FILENAME), jordanNetwork);

		// Backprop section

		// final BasicNetwork feedforwardNetwork =
		// JordanLoto.createFeedforwardNetwork();

		// final double feedforwardError =
		// JordanLoto.trainNetwork("Feedforward",feedforwardNetwork,
		// trainingSet);

		// Save feedforward Network
		// EncogDirectoryPersistence.saveObject(new
		// File(Config.JORDANFEEDFORWARD_FILENAME), feedforwardNetwork);

		log.debug("Best error rate with Jordan Network: " + jordanError);
		// log.debug("Best error rate with Feedforward Network: " +
		// feedforwardError);
		log.debug("Jordan will perform only marginally better than feedforward."
				+ "\nThe more output neurons, the better performance a Jordan will give.");

		return jordanNetwork;
	}

	/*
	 * Continue training from the last saved network
	 */
	public BasicNetwork loadAndContinueTrain(int sourceTrainData,
			BasicNetwork jordanNetwork) {

		if (jordanNetwork == null) {
			log.debug("Loading JORDAN network");
			jordanNetwork = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.JORDAN_FILENAME));
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

		double e = jordanNetwork.calculateError(trainingSet);
		log.debug("Loaded Jordan network's error for previous train set is: "
				+ e);

		final double jordanError = JordanLoto.trainNetwork("Jordan",
				jordanNetwork, trainingSet);

		// Save Jordan Network
		EncogDirectoryPersistence.saveObject(new File(
				ConfigLoto.JORDAN_FILENAME), jordanNetwork);

		// Backprop section

		// final BasicNetwork feedforwardNetwork =
		// JordanLoto.createFeedforwardNetwork();

		// final double feedforwardError =
		// JordanLoto.trainNetwork("Feedforward",feedforwardNetwork,
		// trainingSet);

		// Save feedforward Network
		// EncogDirectoryPersistence.saveObject(new
		// File(Config.JORDANFEEDFORWARD_FILENAME), feedforwardNetwork);

		log.debug("Best error rate with Jordan Network: " + jordanError);
		// log.debug("Best error rate with Feedforward Network: " +
		// feedforwardError);
		log.debug("Jordan will perform only marginally better than feedforward."
				+ "\nThe more output neurons, the better performance a Jordan will give.");

		return jordanNetwork;
	}

	public void loadAndEvaluate(BasicNetwork jordanNetwork) {

		if (jordanNetwork == null) {
			log.debug("Loading JORDAN network");
			jordanNetwork = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.JORDAN_FILENAME));
		}

		final MLDataSet testSet = new SQLNeuralDataSet(ConfigLoto.TESTSQL,
				ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
				ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL, ConfigLoto.SQL_UID,
				ConfigLoto.SQL_PWD);
		if (testSet.size() > 0) {
			double e = jordanNetwork.calculateError(testSet);
			log.debug("Loaded Jordan network's error for test set is: " + e);

			// test the neural network
			log.debug("****     Neural Network Results:");
			ConfigLoto.evaluate(jordanNetwork, testSet);
		} else {
			log.debug("Test set is empty");
		}
	}

	public static void main(String[] args) {
		try {
			String arg1 = null;
			if (args.length != 0) {
				arg1 = args[0]; // means load eg file
			}

			// load a properties file from class path, inside static method
			prop.load(JordanLoto.class.getClassLoader().getResourceAsStream(
					"config.properties"));

			// get the property value and print it out
			/*
			 * System.out.println(prop.getProperty("database"));
			 */

			JordanLoto program = new JordanLoto();

			BasicNetwork jordanNetwork = null;
			File networkFile = null;

			if (arg1 != null) {
				// use the previous saved eg file so no training
				try {
					networkFile = new File(ConfigLoto.JORDAN_FILENAME);
					if (!networkFile.exists()) {
						log.debug("Can't read Jordan eg file: "
								+ networkFile.getAbsolutePath());
						jordanNetwork = program
								.trainAndSave(ConfigLoto.DATASOURCESQL);
					} else {
						jordanNetwork = program.loadAndContinueTrain(ConfigLoto.DATASOURCESQL,jordanNetwork);
					}

				} catch (Throwable t) {
					t.printStackTrace();
					jordanNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);
				} finally {
				}
				if (jordanNetwork == null) {
					jordanNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);
				}
				program.loadAndEvaluate(jordanNetwork);
			} else {
				jordanNetwork = program.trainAndSave(ConfigLoto.DATASOURCESQL);
				program.loadAndEvaluate(jordanNetwork);
			}
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			Encog.getInstance().shutdown();
		}

	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}
}
