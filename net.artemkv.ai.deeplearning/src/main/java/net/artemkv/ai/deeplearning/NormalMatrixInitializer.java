package net.artemkv.ai.deeplearning;

import java.util.Random;

public class NormalMatrixInitializer implements MatrixInitializer {
    private static final Random random = new Random();
    private int incomingLinks;

    public NormalMatrixInitializer(int incomingLinks) {
        if (incomingLinks <= 0) {
            throw new IllegalArgumentException("incomingLinks");
        }

        this.incomingLinks = incomingLinks;
    }

    public float getValue(int row, int column) {
        boolean isPositive = random.nextBoolean();
        float nextAbsoluteValue = random.nextFloat() / (float) Math.sqrt(incomingLinks);
        return isPositive ? nextAbsoluteValue : -nextAbsoluteValue;
    }
}
