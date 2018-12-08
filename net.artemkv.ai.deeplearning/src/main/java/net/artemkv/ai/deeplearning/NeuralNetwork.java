package net.artemkv.ai.deeplearning;

public final class NeuralNetwork {
    private static final float LEARNING_RATE = 0.1F;

    private final int depth;
    private final int inputWidth;
    private final int hiddenWidth;
    private final int outputWidth;
    private final Matrix[] weights;

    /**
     * Initializes a new instance of neural network.
     *
     * @param depth       The depth of the neural network, not counting input layer.
     * @param inputWidth  The width of the input layer.
     * @param hiddenWidth The width of the hidden layer.
     * @param outputWidth The width of the output layer.
     */
    public NeuralNetwork(int depth, int inputWidth, int hiddenWidth, int outputWidth) {
        if (depth < 2) {
            throw new IllegalArgumentException(
                "at least 1 hidden layer and 1 output layer is required");
        }
        if (inputWidth <= 0) {
            throw new IllegalArgumentException("inputWidth");
        }
        if (hiddenWidth <= 0) {
            throw new IllegalArgumentException("hiddenWidth");
        }
        if (outputWidth <= 0) {
            throw new IllegalArgumentException("outputWidth");
        }

        this.depth = depth;
        this.inputWidth = inputWidth;
        this.hiddenWidth = hiddenWidth;
        this.outputWidth = outputWidth;

        weights = new Matrix[depth];

        initialize();
    }

    private void initialize() {
        for (int i = 0; i < depth; i++) {
            if (i == 0) {
                MatrixInitializer initializer = new NormalMatrixInitializer(inputWidth);
                weights[i] = new Matrix(hiddenWidth, inputWidth, initializer);
            } else if (i == depth - 1) {
                MatrixInitializer initializer = new NormalMatrixInitializer(hiddenWidth);
                weights[i] = new Matrix(outputWidth, hiddenWidth, initializer);
            } else {
                MatrixInitializer initializer = new NormalMatrixInitializer(hiddenWidth);
                weights[i] = new Matrix(hiddenWidth, hiddenWidth, initializer);
            }
        }
    }

    public float[] classify(float[] input) {
        if (input.length != inputWidth) {
            throw new IllegalArgumentException(
                String.format("expected input of length %d", inputWidth));
        }
        return getOutput(input);
    }

    public void train(float[] input, float[] expectedOutput) {
        if (input.length != inputWidth) {
            throw new IllegalArgumentException(
                String.format("expected input of length %d", inputWidth));
        }
        if (expectedOutput.length != outputWidth) {
            throw new IllegalArgumentException(
                String.format("expected output of length %d", outputWidth));
        }

        // Calculate outputs
        float[][] layerOutputs = new float[depth][];
        float[] layerOutput = null;
        float[] layerInput = input;
        for (int layer = 0; layer < depth; layer++) {
            // Calculate new value based on input and weights
            layerOutput = weights[layer].multiplyVector(layerInput);
            // Apply activation function
            for (int i = 0; i < layerOutput.length; i++) {
                layerOutput[i] = ActivationFunctions.sigmoid(layerOutput[i]);
            }
            // Save layer
            layerOutputs[layer] = layerOutput;
            // Old out is a new in
            layerInput = layerOutput;
        }

        // Calculate error on the outer layer
        float[] errors = new float[layerOutput.length];
        for (int i = 0; i < errors.length; i++) {
            errors[i] = expectedOutput[i] - layerOutput[i];
        }

        // Backpropagate the error and adjust weights
        for (int layer = depth - 1; layer >= 0; layer--) {
            // Calculate errors for the previous) layer before we update weights
            float[] propagatedErrors = weights[layer].invertAndMultiplyVector(errors);

            // Adjust weights
            final float[] currentErrors = errors;
            final float[] currentOutputs = layerOutputs[layer];
            final float[] previousOutputs = layer > 0 ? layerOutputs[layer - 1] : input;
            Matrix delta = new Matrix(
                weights[layer].getRows(),
                weights[layer].getColumns(),
                new MatrixInitializer() {
                    public float getValue(int row, int column) {
                        return LEARNING_RATE *
                            currentErrors[row] *
                            currentOutputs[row] * (1 - currentOutputs[row]) *
                            previousOutputs[column];
                    }
                });
            weights[layer].applyDelta(delta);

            // Propagate errors
            errors = propagatedErrors;
        }
    }

    private float[] getOutput(float[] input) {
        float[] layerOutput = null;
        float[] layerInput = input;
        for (int layer = 0; layer < depth; layer++) {
            // Calculate new value based on input and weights
            layerOutput = weights[layer].multiplyVector(layerInput);
            // Apply activation function
            for (int i = 0; i < layerOutput.length; i++) {
                layerOutput[i] = ActivationFunctions.sigmoid(layerOutput[i]);
            }
            // Old out is a new in
            layerInput = layerOutput;
        }
        return layerOutput;
    }
}
