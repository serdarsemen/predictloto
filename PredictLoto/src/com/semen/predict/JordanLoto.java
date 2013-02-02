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
//import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.JordanPattern;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

import java.io.File;
import java.text.DecimalFormat;
import java.text.NumberFormat;
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
	private static final long serialVersionUID = 1L;
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

		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			log.debug("Training " + what + ", Epoch #" + epoch + " Error:"
					+ trainMain.getError());
			epoch++;
		}
		return trainMain.getError();
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

	public void loadAndEvaluate(BasicNetwork network) {

		if (network == null) {
			log.debug("Loading JORDAN network");
			network = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.JORDAN_FILENAME));
		}

		final MLDataSet testSet = new SQLNeuralDataSet(ConfigLoto.TESTSQL,
				ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
				ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL, ConfigLoto.SQL_UID,
				ConfigLoto.SQL_PWD);

		double e = network.calculateError(testSet);
		log.debug("Loaded network's error is(should be same as above): " + e);

		// test the neural network
		log.debug("****     Neural Network Results:");
		EncogUtility.evaluate(network, testSet);
	}

	public static void main(String[] args) {
		try {

			// load a properties file from class path, inside static method
			prop.load(JordanLoto.class.getClassLoader().getResourceAsStream(
					"config.properties"));

			// get the property value and print it out
			/*
			 * System.out.println(prop.getProperty("database"));
			 * System.out.println(prop.getProperty("dbuser"));
			 * System.out.println(prop.getProperty("dbpassword"));
			 */

			JordanLoto program = new JordanLoto();
			// 0 from MSSQL 1 from .csv text file
			BasicNetwork jordanNetwork = program.trainAndSave(0);
			program.loadAndEvaluate(jordanNetwork);
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			Encog.getInstance().shutdown();
		}

	}
}
