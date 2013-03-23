package com.semen.predict;

/*
 * Predict Loto 1.0
 * Using Encog(tm) Java Examples v3.2
 * @author serdar semen
 * @version 1.0
 */

import java.io.File;
import java.util.Properties;

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
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.platformspecific.j2se.data.SQLNeuralDataSet;
import org.encog.util.csv.CSVFormat;

import com.semen.util.MySQLUtil;
import org.encog.util.simple.TrainingSetUtil;
import org.encog.util.Format;

/**
 * Implement an Elman style neural network with Encog. This network attempts to
 * predict the next value in Loto
 * 
 * @author serdar semen
 * 
 */
public class ElmanLoto {
	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(ElmanLoto.class); // .getName());
	private static final long serialVersionUID = -343434332L;

	private static Properties prop = new Properties();

	/*
	 * For each file, you'll need a separate Logger. private static Logger log =
	 * Logger.getLogger( JordanLoto.class ) private static Logger connectionsLog
	 * = Logger.getLogger( "connections." + JordanLoto.class.getName() ) private
	 * static Logger stacktracesLog = Logger.getLogger( "stacktraces." +
	 * JordanLoto.class.getName() ) private static Logger httpLog =
	 * Logger.getLogger( "http." + JordanLoto.class.getName() )
	 */

	static BasicNetwork createElmanNetwork() {
		// construct an Elman type network
		ElmanPattern pattern = new ElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(ConfigLoto.INPUT_SIZE);
		pattern.addHiddenLayer(ConfigLoto.ELMANHIDDENNEURONSIZE);
		pattern.setOutputNeurons(ConfigLoto.IDEAL_SIZE);
		log.debug("LO_WEEKNO= " + ConfigLoto.LO_WEEKNO);
		log.debug("HI_WEEKNO= " + ConfigLoto.HI_WEEKNO);

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
		log.debug("LO_WEEKNO= " + ConfigLoto.LO_WEEKNO);
		log.debug("HI_WEEKNO= " + ConfigLoto.HI_WEEKNO);

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
		// EncogUtility.trainToError(trainMain, ConfigLoto.ELMANDESIREDERROR);

		int epoch = 1;
		double trainError = 1.0;
		double prevtrainError = 1.0;
		double sameErrorCount = 0;

		double desired_Error = ConfigLoto.ELMANDESIREDERROR;
		String str_TargetError = Format.formatDouble(desired_Error, 4);
		while (   //(!stop.shouldStop())	&& 
				(trainError > desired_Error)
				&& (sameErrorCount < ConfigLoto.NEATEPOCHEXITCOUNTER)) {
			if (prevtrainError == trainError) {
				sameErrorCount++;
			} else {
			//	log.debug("SameErrorCount=0");
				sameErrorCount = 0;
			}
			prevtrainError = trainError;
			trainMain.iteration();
			trainError = trainMain.getError();
			log.debug(what + "  #" + epoch + " Err= "
					+ Format.formatDouble(trainError, 4) + " Target Err= "
					+ str_TargetError);
			if ((epoch % ConfigLoto.EPOCHSAVEINTERVAL) == 0) {
				log.debug("Saving " + what + ", Epoch #" + epoch);
				// Save feedforward Network
				if (what.equals("Elman"))
					EncogDirectoryPersistence.saveObject(new File(
							ConfigLoto.ELMAN_FILENAME), network);
				else
					// Save Elman Network
					EncogDirectoryPersistence.saveObject(new File(
							ConfigLoto.ELMANFEEDFORWARD_FILENAME), network);
			}
			epoch++;
		}
		trainMain.finishTraining();
		
		ConfigLoto.INSERTSAYISALPREDICTPART1 = "\"Elman\"," + desired_Error + ",0,0,";
				
		// not yet supported
		// trainMain.dump(new File(ConfigLoto.ELMAN_DUMPFILENAME));
		return trainError;
	}

	public BasicNetwork trainAndSave(int sourceTrainData) {

		MLDataSet trainingSet = null;
		if (sourceTrainData == ConfigLoto.DATASOURCESQL)

			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL,
					MySQLUtil.SQL_UID, MySQLUtil.SQL_PWD);

		else if (sourceTrainData == ConfigLoto.DATASOURCECSV)
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
		/*
		 * final BasicNetwork feedforwardNetwork = ElmanLoto
		 * .createFeedforwardNetwork();
		 * 
		 * final double feedforwardError = ElmanLoto.trainNetwork("Feedforward",
		 * feedforwardNetwork, trainingSet);
		 */
		// Save feedforward Network
		// EncogDirectoryPersistence.saveObject(new File(
		// ConfigLoto.ELMANFEEDFORWARD_FILENAME), feedforwardNetwork);

		log.debug("Best error rate with Elman Network: " + elmanError);
		// log.debug("Best error rate with Feedforward Network: "+
		// feedforwardError);
		// log.debug("Elman should be able to get into the 10% range,"
		// + "\nfeedforward should not go below 25%."
		// + "\nThe recurrent Elment net can learn better in this case.");
		// log.debug("If your results are not as good, try rerunning, or perhaps training longer.");

		return elmanNetwork;
	}

	/*
	 * Continue training from the last saved network
	 */
	public BasicNetwork loadAndContinueTrain(int sourceTrainData,
			BasicNetwork elmanNetwork) {

		if (elmanNetwork == null) {
			log.debug("Loading ELMAN network");
			elmanNetwork = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.ELMAN_FILENAME));
		}

		MLDataSet trainingSet = null;
		if (sourceTrainData == ConfigLoto.DATASOURCESQL)

			trainingSet = new SQLNeuralDataSet(ConfigLoto.TRAINSQL,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
					MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL,
					MySQLUtil.SQL_UID, MySQLUtil.SQL_PWD);

		else if (sourceTrainData == ConfigLoto.DATASOURCECSV)
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);
		else
			trainingSet = TrainingSetUtil.loadCSVTOMemory(
					CSVFormat.DECIMAL_COMMA, ConfigLoto.trainCSVFile, true,
					ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE);
		if (elmanNetwork != null) {
			double e = elmanNetwork.calculateError(trainingSet);
			log.debug("Loaded Elman  network's error for previous train set is: "
					+ e);

			final double elmanError = ElmanLoto.trainNetwork("Elman",
					elmanNetwork, trainingSet);

			// Save Elman Network
			EncogDirectoryPersistence.saveObject(new File(
					ConfigLoto.ELMAN_FILENAME), elmanNetwork);

			// Backprop section
			/*
			 * final BasicNetwork feedforwardNetwork = ElmanLoto
			 * .createFeedforwardNetwork();
			 * 
			 * final double feedforwardError =
			 * ElmanLoto.trainNetwork("Feedforward", feedforwardNetwork,
			 * trainingSet);
			 */
			// Save feedforward Network
			// EncogDirectoryPersistence.saveObject(new File(
			// ConfigLoto.ELMANFEEDFORWARD_FILENAME), feedforwardNetwork);

			log.debug("Best error rate with Elman Network: " + elmanError);
			// log.debug("Best error rate with Feedforward Network: "+
			// feedforwardError);
		} else {
			log.debug("ELMAN network is NULL");
		}

		return elmanNetwork;
	}

	public void loadAndEvaluate(BasicNetwork elmanNetwork) {

		if (elmanNetwork == null) {
			log.debug("Loading ELMAN network");
			elmanNetwork = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(ConfigLoto.ELMAN_FILENAME));
		}

		final MLDataSet testSet = new SQLNeuralDataSet(ConfigLoto.TESTSQL,
				ConfigLoto.INPUT_SIZE, ConfigLoto.IDEAL_SIZE,
				MySQLUtil.SQL_DRIVER, MySQLUtil.SQL_URL, MySQLUtil.SQL_UID,
				MySQLUtil.SQL_PWD);
		if (testSet.size() > 0) {
			double e = elmanNetwork.calculateError(testSet);
			log.debug("Loaded Elman network's error for test set is: " + e);

			// test the neural network
			log.debug("**** Elman Neural Network Results:");
			ConfigLoto.evaluate(elmanNetwork, testSet);
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
			prop.load(ElmanLoto.class.getClassLoader().getResourceAsStream(
					"config.properties"));

			// get the property value
			/*
			 * System.out.println(prop.getProperty("database"));
			 */

			ElmanLoto program = new ElmanLoto();
			BasicNetwork elmanNetwork = null;
			File networkFile = null;

			if (arg1 != null) {
				// use the previous saved eg file so no training
				try {
					networkFile = new File(ConfigLoto.ELMAN_FILENAME);
					if (!networkFile.exists()) {
						log.debug("Can't read Elman eg file: "
								+ networkFile.getAbsolutePath());
						elmanNetwork = program
								.trainAndSave(ConfigLoto.DATASOURCESQL);
					} else {
						elmanNetwork = program.loadAndContinueTrain(
								ConfigLoto.DATASOURCESQL, elmanNetwork);
					}
				} catch (Throwable t) {
					t.printStackTrace();
					elmanNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);
				} finally {
				}
				if (elmanNetwork == null) {
					elmanNetwork = program
							.trainAndSave(ConfigLoto.DATASOURCESQL);
				}

				program.loadAndEvaluate(elmanNetwork);
			} else {
				elmanNetwork = program.trainAndSave(ConfigLoto.DATASOURCESQL);
				program.loadAndEvaluate(elmanNetwork);
			}
		} catch (Throwable t) {
			t.printStackTrace();
		} finally {
			double estimatedTimeMin = (System.nanoTime() - startTime) / 60000000000.0;
			log.debug("Elapsed Time  = " + ConfigLoto.round2(estimatedTimeMin)
					+ " (min) ");
			Encog.getInstance().shutdown();
		}
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}
}
