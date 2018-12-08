package net.artemkv.ai.deeplearning.training;

public class TrainingInput {
    private static final float Y = 0.99F;
    private static final float N = 0.01F;
    private static final float[] LABEL_0 = { Y, N, N, N, N, N, N, N, N, N };
    private static final float[] LABEL_1 = { N, Y, N, N, N, N, N, N, N, N };
    private static final float[] LABEL_2 = { N, N, Y, N, N, N, N, N, N, N };
    private static final float[] LABEL_3 = { N, N, N, Y, N, N, N, N, N, N };
    private static final float[] LABEL_4 = { N, N, N, N, Y, N, N, N, N, N };
    private static final float[] LABEL_5 = { N, N, N, N, N, Y, N, N, N, N };
    private static final float[] LABEL_6 = { N, N, N, N, N, N, Y, N, N, N };
    private static final float[] LABEL_7 = { N, N, N, N, N, N, N, Y, N, N };
    private static final float[] LABEL_8 = { N, N, N, N, N, N, N, N, Y, N };
    private static final float[] LABEL_9 = { N, N, N, N, N, N, N, N, N, Y };
    private static final float[][] labels = new float[][]{
        LABEL_0, LABEL_1, LABEL_2, LABEL_3, LABEL_4,
        LABEL_5, LABEL_6, LABEL_7, LABEL_8, LABEL_9
    };

    private final SensorInput input;
    private final int value;

    public TrainingInput(SensorInput input, int value) {
        if (input == null) {
            throw new IllegalArgumentException("input");
        }
        if (value < 0 || value > 9) {
            throw new IllegalArgumentException("value is expected to be in 0..9 range");
        }

        this.input = input;
        this.value = value;
    }

    public float[] getScaledInput() {
        int[] data = input.getData();
        float max = (float) data[0];
        float[] scaled = new float[data.length];
        for (int i = 0; i < scaled.length; i++) {
            scaled[i] = (float) data[i];
            if (scaled[i] > max) {
                max = scaled[i];
            }
        }
        for (int i = 0; i < scaled.length; i++) {
            scaled[i] = scaled[i] / max * 0.98F + 0.01F;
        }
        return scaled;
    }

    public float[] getLabel() {
        return labels[value];
    }
}
