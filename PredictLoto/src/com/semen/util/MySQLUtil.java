package com.semen.util;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;



/**
 * Utilities for using the My SQL (MySQL) engine.
 */
public final class MySQLUtil {

	/**
	 * The driver to use for MySQL
	 */
	public final static String SQL_DRIVER = "com.mysql.jdbc.Driver";
	

	/**
	 * The URL to use for the  database.
	 */
	public final static String SQL_URL = "jdbc:mysql://localhost:3306/loto";
	

	/**
	 * The user id to use for the  database.
	 */
	public final static String SQL_UID = "root";
	
	/**
	 * The password to use for the memory database.
	 */
	public final static String SQL_PWD = "serdar";

	
	/**
	 * The dialect to use for the memory database.
	 */
	public static final String DIALECT = "org.hibernate.dialect.MySQLDialect";

	/**
	 * @return A connection to the memory database.
	 * @throws SQLException
	 *             An SQL error.
	 */
	public static Connection getConnection() throws SQLException {
		final Properties props = new Properties();
		props.put("user", SQL_UID);
		props.put("password", SQL_PWD);

		return DriverManager.getConnection(SQL_URL, props);
	}

	/**
	 * Load the driver for the memory database.
	 * 
	 * @throws InstantiationException
	 *             Database error.
	 * @throws IllegalAccessException
	 *             Database error.
	 * @throws ClassNotFoundException
	 *             Database error.
	 */
	public static void loadDriver() throws InstantiationException,
			IllegalAccessException, ClassNotFoundException {
		Class.forName(SQL_DRIVER);
	}

	/**
	 * Shutdown the database. Not currently implemented.
	 */
	public static void shutdown() {

	}

	public static void ExecuteSQLCommand(String sqlStr) throws Exception {
		MySQLUtil.loadDriver();

		Connection conn = MySQLUtil.getConnection();

		conn.setAutoCommit(true);

		Statement s = conn.createStatement();

		s.execute(sqlStr);
		MySQLUtil.shutdown();

	}

	/**
	 * Private constructor.
	 */
	private MySQLUtil() {
	}
}
