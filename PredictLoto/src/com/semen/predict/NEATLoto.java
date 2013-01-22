/*
 */
package com.semen.predict;

import java.io.File;

import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.ml.data.MLDataSet;

import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATTraining;
import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.csv.CSVFormat;
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
	public static final Logger log = Logger.getLogger(NEATLoto.class); //.getName());
	private static final long serialVersionUID = 3L;

	public NEATNetwork trainAndSave(int sourceTrainData) {

		MLDataSet trainingSet = null;
		if (sourceTrainData == 0)

			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					ConfigLoto.SQL_DRIVER, ConfigLoto.SQL_URL,
					ConfigLoto.SQL_UID, ConfigLoto.SQL_PWD);

		else if (sourceTrainData == 1)
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true, ConfigLoto.INPUT_SIZE,
					ConfigLoto.IDEAL_SIZE);
		else
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true, ConfigLoto.INPUT_SIZE,
					ConfigLoto.IDEAL_SIZE);

		NEATPopulation pop = new NEATPopulation(ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE, ConfigLoto.NEATPOPULATIONSIZE);
		CalculateScore score = new TrainingSetScore(trainingSet);

		// train the neural network
		final NEATTraining train = new NEATTraining(score, pop);

		EncogUtility.trainToError(train, ConfigLoto.NEATDESIREDERROR);

		NEATNetwork network = (NEATNetwork) train.getMethod();

		// Save NEAT Network  
		// TODO : pop saved but neatnetwork did not work supported ?
		EncogDirectoryPersistence.saveObject(
				new File(ConfigLoto.NEAT_FILENAME), pop); //network ?? 
		return network;
	}

	public void loadAndEvaluate(NEATNetwork network) {

		if (network == null) {
			log.debug("Loading NEAT network");
			network = (NEATNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.NEAT_FILENAME));

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
			NEATLoto program = new NEATLoto();
			// 0 from MSSQL 1 from .csv text file
			NEATNetwork neatNetwork = program.trainAndSave(1);
			program.loadAndEvaluate(neatNetwork);
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			Encog.getInstance().shutdown();
		}

	}
}
