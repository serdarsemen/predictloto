package com.semen.predict;

/*
 * Predict Loto 1.0
 * Using Encog(tm) Java Examples v3.2
 * @author serdar semen
 * @version 1.0
 */

import java.io.File;

import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

//import com.semen.util.LotoConfig;

/**
 * Implement an Elman style neural network with Encog. This network attempts to
 * predict the next value in Loto
 * 
 * @author serdar semen
 * 
 */
public class ElmanLoto {
	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(ElmanLoto.class); //.getName());
	private static final long serialVersionUID = 2L;

	static BasicNetwork createElmanNetwork() {
		// construct an Elman type network
		ElmanPattern pattern = new ElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(ConfigLoto.INPUT_SIZE);
		pattern.addHiddenLayer(ConfigLoto.ELMANHIDDENNEURONSIZE);
		pattern.setOutputNeurons(ConfigLoto.IDEAL_SIZE);
		return (BasicNetwork) pattern.generate();
	}

	static BasicNetwork createFeedforwardNetwork() {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());// or
		// ActivationTANH
		pattern.setInputNeurons(ConfigLoto.INPUT_SIZE);
		pattern.addHiddenLayer(ConfigLoto.FEEDFORWARDHIDDENNEURONSIZE);
		pattern.setOutputNeurons(ConfigLoto.IDEAL_SIZE);
		return (BasicNetwork) pattern.generate();
	}

	public static double trainNetwork(final String what,
			final BasicNetwork network, final MLDataSet trainingSet) {
		// train the neural network
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
			log.debug("Training " + what + ", Epoch #" + epoch
					+ " Error:" + trainMain.getError());
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

		final BasicNetwork elmanNetwork = ElmanLoto.createElmanNetwork();

		final double elmanError = ElmanLoto.trainNetwork("Elman", elmanNetwork,
				trainingSet);

		// Save Elman Network
		EncogDirectoryPersistence.saveObject(
				new File(ConfigLoto.ELMAN_FILENAME), elmanNetwork);

		// Backprop section

		// final BasicNetwork feedforwardNetwork =
		// ElmanLoto.createFeedforwardNetwork();

		// final double feedforwardError = ElmanLoto.trainNetwork("Feedforward",
		// feedforwardNetwork, trainingSet);

		// Save feedforward Network
		// EncogDirectoryPersistence.saveObject(new
		// File(ELMANFEEDFORWARD_FILENAME), feedforwardNetwork);

		log.debug("Best error rate with Elman Network: " + elmanError);
		// log.debug("Best error rate with Feedforward Network: " +
		// feedforwardError);
		log.debug("Elman should be able to get into the 10% range,"
				+ "\nfeedforward should not go below 25%."
				+ "\nThe recurrent Elment net can learn better in this case.");
		log.debug("If your results are not as good, try rerunning, or perhaps training longer.");

		return elmanNetwork;
	}

	public void loadAndEvaluate(BasicNetwork network) {

		if (network == null) {
			log.debug("Loading ELMAN network");
			network = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.ELMAN_FILENAME));
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
			ElmanLoto program = new ElmanLoto();
			// 0 from MSSQL 1 from .csv text file
			BasicNetwork elmanNetwork = program.trainAndSave(1);
			program.loadAndEvaluate(elmanNetwork);
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			Encog.getInstance().shutdown();
		}
	}
}
