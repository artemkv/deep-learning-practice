package net.artemkv.ai.deeplearning.training;

public class SensorInput {
    private final int[] data;

    public SensorInput(int[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data");
        }
        this.data = data;
    }

    public int[] getData() {
        return data;
    }
}
