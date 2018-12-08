package net.artemkv.ai.deeplearning;

import java.util.Random;

class RandomMatrixInitializer implements MatrixInitializer {
    private static final Random random = new Random();

    public float getValue(int row, int column) {
        boolean isPositive = random.nextBoolean();
        return isPositive ? random.nextFloat() : -random.nextFloat();
    }
}
