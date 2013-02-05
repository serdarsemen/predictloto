/*
 * */
package com.semen.predict;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.apache.log4j.Priority;

/**
 * Basic config info for the Predict Loto.
 * 
 * @author serdar semen
 * 
 */
public class ConfigLoto {
	
	/**
	 * The base directory that all of the data for this example is stored in.
	 */
	private static final String basePathStr = "d:\\sayisalAi";

	/* Get actual class name to be printed on */
	public static final Logger log = Logger.getLogger(ConfigLoto.class);//			.getName());
	//private static final NumberFormat nf = new DecimalFormat("0.00");
	//private static final long serialVersionUID = 4L;
	public Properties props;

	public final static int INPUT_SIZE = 49;
	public final static int IDEAL_SIZE = 49;

	public static int JORDANHIDDENNEURONSIZE = 180;
	public static int ELMANHIDDENNEURONSIZE = 180;
	public static int FEEDFORWARDHIDDENNEURONSIZE = 180;
	
	// 0 from MSSQL 1 from .csv text file
	public static int DATASOURCETYPE=0;
	
	// Neat
	public static int NEATPOPULATIONSIZE = 1000; // 1000
	public static double NEATPOPULATIONDENSITY= 0.0; //1.0
	public static double NEATDESIREDERROR = 0.10; // 0.01 En çabuk 0.24 0.32 olabiliyor  0.1083
	public static double JORDANDESIREDERROR = 0.1205;
	public static double ELMANDESIREDERROR = 0.107;
	public static double FEEDFORWARDDESIREDERROR = 0.14;
	// Neural Simulated Annealing
	public static double SIMANNEAL_STARTTEMP = 10.0; // The starting
														// temperature.
	public static double SIMANNEAL_STOPTEMP = 2.0; //2.0 The ending temperature.
	public static int SIMANNEAL_CYCLES = 300; // 100 The number of cycles in a
												// training iteration.
	// Backpropagation
	public static double BPROP_LEARNRATE = 0.00001; // The rate at which the
													// weight matrix will be
													// adjusted based on
													// learning.
	public static double BPROP_MOMENTUM = 0.05;// 0.0 The influence that previous
												// iteration's training deltas
												// will have on the current
												// iteration.

	public static final String trainCSVFile = basePathStr + "\\inp49.csv";
	public static final String testCSVFile = basePathStr + "\\test49.csv";

	public static final String JORDAN_FILENAME = basePathStr
			+ "\\JordanLoto.eg";
	public static final String JORDANFEEDFORWARD_FILENAME = basePathStr
			+ "\\JordanFeedForwardLoto.eg";
	public static final String NEAT_FILENAME = basePathStr + "\\NeatLoto.eg";
	public static final String ELMAN_FILENAME = basePathStr + "\\ElmanLoto.eg";
	public static final String ELMANFEEDFORWARD_FILENAME = basePathStr
			+ "\\ElmanFeedForwardLoto.eg";

	public final static int TRAIN_SIZE = 840; // 

	public final static String SELECTSQL = "SELECT `lotoresults`.`input1`,"
			+ "`lotoresults`.`input2`," + "`lotoresults`.`input3`,"
			+ "`lotoresults`.`input4`," + "`lotoresults`.`input5`,"
			+ "`lotoresults`.`input6`," + "`lotoresults`.`input7`,"
			+ "`lotoresults`.`input8`," + "`lotoresults`.`input9`,"
			+ "`lotoresults`.`input10`," + "`lotoresults`.`input11`,"
			+ "`lotoresults`.`input12`," + "`lotoresults`.`input13`,"
			+ "`lotoresults`.`input14`," + "`lotoresults`.`input15`,"
			+ "`lotoresults`.`input16`," + "`lotoresults`.`input17`,"
			+ "`lotoresults`.`input18`," + "`lotoresults`.`input19`,"
			+ "`lotoresults`.`input20`," + "`lotoresults`.`input21`,"
			+ "`lotoresults`.`input22`," + "`lotoresults`.`input23`,"
			+ "`lotoresults`.`input24`," + "`lotoresults`.`input25`,"
			+ "`lotoresults`.`input26`," + "`lotoresults`.`input27`,"
			+ "`lotoresults`.`input28`," + "`lotoresults`.`input29`,"
			+ "`lotoresults`.`input30`," + "`lotoresults`.`input31`,"
			+ "`lotoresults`.`input32`," + "`lotoresults`.`input33`,"
			+ "`lotoresults`.`input34`," + "`lotoresults`.`input35`,"
			+ "`lotoresults`.`input36`," + "`lotoresults`.`input37`,"
			+ "`lotoresults`.`input38`," + "`lotoresults`.`input39`,"
			+ "`lotoresults`.`input40`," + "`lotoresults`.`input41`,"
			+ "`lotoresults`.`input42`," + "`lotoresults`.`input43`,"
			+ "`lotoresults`.`input44`," + "`lotoresults`.`input45`,"
			+ "`lotoresults`.`input46`," + "`lotoresults`.`input47`,"
			+ "`lotoresults`.`input48`," + "`lotoresults`.`input49`,"
			+ "`lotoresults`.`ideal1`," + "`lotoresults`.`ideal2`,"
			+ "`lotoresults`.`ideal3`," + "`lotoresults`.`ideal4`,"
			+ "`lotoresults`.`ideal5`," + "`lotoresults`.`ideal6`,"
			+ "`lotoresults`.`ideal7`," + "`lotoresults`.`ideal8`,"
			+ "`lotoresults`.`ideal9`," + "`lotoresults`.`ideal10`,"
			+ "`lotoresults`.`ideal11`," + "`lotoresults`.`ideal12`,"
			+ "`lotoresults`.`ideal13`," + "`lotoresults`.`ideal14`,"
			+ "`lotoresults`.`ideal15`," + "`lotoresults`.`ideal16`,"
			+ "`lotoresults`.`ideal17`," + "`lotoresults`.`ideal18`,"
			+ "`lotoresults`.`ideal19`," + "`lotoresults`.`ideal20`,"
			+ "`lotoresults`.`ideal21`," + "`lotoresults`.`ideal22`,"
			+ "`lotoresults`.`ideal23`," + "`lotoresults`.`ideal24`,"
			+ "`lotoresults`.`ideal25`," + "`lotoresults`.`ideal26`,"
			+ "`lotoresults`.`ideal27`," + "`lotoresults`.`ideal28`,"
			+ "`lotoresults`.`ideal29`," + "`lotoresults`.`ideal30`,"
			+ "`lotoresults`.`ideal31`," + "`lotoresults`.`ideal32`,"
			+ "`lotoresults`.`ideal33`," + "`lotoresults`.`ideal34`,"
			+ "`lotoresults`.`ideal35`," + "`lotoresults`.`ideal36`,"
			+ "`lotoresults`.`ideal37`," + "`lotoresults`.`ideal38`,"
			+ "`lotoresults`.`ideal39`," + "`lotoresults`.`ideal40`,"
			+ "`lotoresults`.`ideal41`," + "`lotoresults`.`ideal42`,"
			+ "`lotoresults`.`ideal43`," + "`lotoresults`.`ideal44`,"
			+ "`lotoresults`.`ideal45`," + "`lotoresults`.`ideal46`,"
			+ "`lotoresults`.`ideal47`," + "`lotoresults`.`ideal48`,"
			+ "`lotoresults`.`ideal49` ";

	public final static String TRAINSQL = SELECTSQL + " FROM lotoresults "
			+ " WHERE weekid<=" + TRAIN_SIZE + " ORDER BY weekid";

	public final static String TESTSQL = SELECTSQL + " FROM lotoresults "
			+ " WHERE weekid>" + TRAIN_SIZE + " ORDER BY weekid";

	public final static String SQL_DRIVER = "com.mysql.jdbc.Driver";
	public final static String SQL_URL = "jdbc:mysql://localhost:3306/loto";
	public final static String SQL_UID = "root";
	public final static String SQL_PWD = "serdar";


	
	
	
	
	
	//
	
	public static final String TRIAL_COUNT = "fitness.function.tmaze.trial.count";
	public static final String REWARD_SWITCH_COUNT = "fitness.function.tmaze.reward.switch.count";
	public static final String REWARD_SWITCH_VARIATION = "fitness.function.tmaze.reward.switch.variation";
	public static final String REWARD_LOW = "fitness.function.tmaze.reward.low";
	public static final String REWARD_HIGH = "fitness.function.tmaze.reward.high";
	public static final String REWARD_LOW_COLOUR = "fitness.function.tmaze.reward.low.colour";
	public static final String REWARD_HIGH_COLOUR = "fitness.function.tmaze.reward.high.colour";
	public static final String REWARD_CRASH = "fitness.function.tmaze.reward.crash";
	public static final String PASSAGE_LENGTH = "fitness.function.tmaze.passage.length";
	public static final String DOUBLE_TMAZE = "fitness.function.tmaze.double";
	
	
	private boolean isDouble;
	private int trialCount, rewardSwitchCount, passageLength;
	private double rewardSwitchVariation;
	private double rewardLow, rewardHigh, rewardCrash, rewardLowColour, rewardHighColour;
	private int[] rewardSwitchTrials, rewardIndexForSwitch;
	private byte[][] map; // The map of the maze, consists of values from TYPE_*, format is [x][y].
	private int startX, startY; // Initial location of agent in map.
	private int[] rewardLocationsX, rewardLocationsY;
/*
	public void init(Properties props) {
		super.init(props);
		isDouble = props.getBooleanProperty(DOUBLE_TMAZE, false);
		trialCount = props.getIntProperty(TRIAL_COUNT, 200);
		rewardSwitchCount = props.getIntProperty(REWARD_SWITCH_COUNT, 3);
		rewardSwitchVariation = props.getDoubleProperty(REWARD_SWITCH_VARIATION, 0.2);
		passageLength = props.getIntProperty(PASSAGE_LENGTH, 3);
		rewardLow = props.getDoubleProperty(REWARD_LOW, 0.1);
		rewardHigh = props.getDoubleProperty(REWARD_HIGH, 1);
		rewardCrash = props.getDoubleProperty(REWARD_CRASH, 0);
		rewardLowColour = props.getDoubleProperty(REWARD_LOW_COLOUR, 0.2);
		rewardHighColour = props.getDoubleProperty(REWARD_HIGH_COLOUR, 1);
	} */
	
	// TODO: make it singleton
    private static final ConfigLoto instance = new ConfigLoto();
    
    private ConfigLoto() {
    	// call init
    }
 
    public static ConfigLoto getInstance() {
        return instance;
    }
	
}
