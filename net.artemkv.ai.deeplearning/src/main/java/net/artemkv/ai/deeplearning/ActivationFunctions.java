package net.artemkv.ai.deeplearning;

final class ActivationFunctions {
    public static float sigmoid(float x) {
        return (float) (1.0F / (1.0F + Math.pow(Math.E, -x)));
    }
}
