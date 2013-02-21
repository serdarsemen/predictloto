package com.semen.predict;

import java.util.Random;

import org.encog.mathutil.IntPair;

public class BoxTrialCase {

	
	public static final int BOUNDS = ConfigLoto.BASE_RESOLUTION - 1;

	private IntPair smallBoxTopLeft;
	private IntPair largeBoxTopLeft;

	private Random rnd;

	public BoxTrialCase(Random theRnd) {
		this.rnd = theRnd;
	}

	public IntPair initTestCase(int largeBoxRelativePos) {
		IntPair[] loc = generateRandomTestCase(largeBoxRelativePos);
		smallBoxTopLeft = loc[0];
		largeBoxTopLeft = (IntPair)loc[1].clone();
		largeBoxTopLeft.add(-1);
		return loc[1];
	}

	public double getPixel(double x, double y) {
		int pixelX = (int) (((x + 1.0) * ConfigLoto.BASE_RESOLUTION) / 2.0);
		int pixelY = (int) (((y + 1.0) * ConfigLoto.BASE_RESOLUTION) / 2.0);

		if (smallBoxTopLeft.getX() == pixelX
				&& smallBoxTopLeft.getY() == pixelY) {
			return 1.0;
		}

		int deltaX = (int) (pixelX - largeBoxTopLeft.getX());
		int deltaY = (int) (pixelY - largeBoxTopLeft.getY());
		return (deltaX > -1 && deltaX < 3 && deltaY > -1 && deltaY < 3) ? 1.0
				: 0.0;
	}

	private IntPair[] generateRandomTestCase(int largeBoxRelativePos) {
		IntPair smallBoxPos = new IntPair(rnd.nextInt(ConfigLoto.BASE_RESOLUTION),
				rnd.nextInt(ConfigLoto.BASE_RESOLUTION));

		IntPair largeBoxPos = (IntPair) smallBoxPos.clone();
		switch (largeBoxRelativePos) {
		case 0:
			largeBoxPos.addX(5);
			break;
		case 1:
			largeBoxPos.addY(5);
			break;
		case 2:
			if (rnd.nextBoolean()) {
				largeBoxPos.add(3, 4);
			} else {
				largeBoxPos.add(4, 3);
			}
			break;
		}

		if (largeBoxPos.getX() > BoxTrialCase.BOUNDS) {
			largeBoxPos.addX(-ConfigLoto.BASE_RESOLUTION);

			if (0 == largeBoxPos.getX()) {
				largeBoxPos.add(1);
			}
		} else if (BoxTrialCase.BOUNDS == largeBoxPos.getX()) {
			largeBoxPos.addX(-1);
		} else if (largeBoxPos.getX() == 0) {
			largeBoxPos.addX(1);
		}

		if (largeBoxPos.getY() > BoxTrialCase.BOUNDS) {
			largeBoxPos.addY(-ConfigLoto.BASE_RESOLUTION);

			if (0 == largeBoxPos.getY()) {
				largeBoxPos.addY(1);
			}
		} else if (BoxTrialCase.BOUNDS == largeBoxPos.getY()) {
			largeBoxPos.addY(-1);
		} else if (0 == largeBoxPos.getY()) {
			largeBoxPos.addY(1);
		}
		return new IntPair[] { smallBoxPos, largeBoxPos };
	}
}
