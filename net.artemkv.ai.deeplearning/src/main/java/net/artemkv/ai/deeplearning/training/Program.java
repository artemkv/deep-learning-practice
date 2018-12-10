package net.artemkv.ai.deeplearning.training;

import net.artemkv.ai.deeplearning.NeuralNetwork;

import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

public class Program {
    public static void main(String[] args) {
        //trainAndSave();
        //loadAndTest();

        classifyMyOwnImages();
    }

    private static void classifyMyOwnImages() {
        final String networkFileName = "/mnist2x100.nn";

        InputStream stream = Program.class.getResourceAsStream(networkFileName);
        NeuralNetwork nn = new NeuralNetwork(stream);
        try {
            classifyMyOwnImage(nn, "/mysamples/0.png");
            classifyMyOwnImage(nn, "/mysamples/1.png");
            classifyMyOwnImage(nn, "/mysamples/2.png");
            classifyMyOwnImage(nn, "/mysamples/3.png");
            classifyMyOwnImage(nn, "/mysamples/4.png");
            classifyMyOwnImage(nn, "/mysamples/5.png");
            classifyMyOwnImage(nn, "/mysamples/6.png");
            classifyMyOwnImage(nn, "/mysamples/7.png");
            classifyMyOwnImage(nn, "/mysamples/8.png");
            classifyMyOwnImage(nn, "/mysamples/9.png");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void classifyMyOwnImage(NeuralNetwork nn, String fileName) throws IOException {
        InputStream stream = Program.class.getResourceAsStream(fileName);
        int[] data = ImageLoader.getData(stream);

        TrainingInput input = new TrainingInput(new SensorInput(data), 0);
        float[] result = nn.classify(input.getScaledInput());
        System.out.println(fileName + " classified as " + getLabel(result));
    }

    private static void trainAndSave() {
        final String networkName = "mnist";
        final String trainingSetFileName = "/mnist_train.csv";
        final String validationSetFileName = "/mnist_test.csv"; // TODO: here we are using test data as a validation set

        // Network configuration
        final int hiddenLayerWidth = 100;
        final int networkDepth = 2;

        NeuralNetwork nn = new NeuralNetwork(networkDepth, 28 * 28, hiddenLayerWidth, 10);

        // Training in epochs
        float targetErrorDelta = 0.001F;
        float validationErrorPrev = Float.MAX_VALUE;
        float validationErrorCur;
        int epoch = 1;
        while(true) {
            // Train
            System.out.println(String.format("Training epoch %d", epoch));
            float trainingError = train(nn, trainingSetFileName);
            System.out.println(String.format("Finished training epoch %d", epoch));

            // Validate
            validationErrorCur = test(nn, validationSetFileName, false);
            System.out.println(String.format("Avg. training error: %f", trainingError));
            System.out.println(String.format("Avg. validation error: %f", validationErrorCur));

            // Check if we trained enough
            if (Math.abs(validationErrorPrev - validationErrorCur) < targetErrorDelta) {
                System.out.println(String.format("%nFinished with error delta: %f",
                    Math.abs(validationErrorPrev - validationErrorCur)));
                System.out.println(String.format("Finished training in %d epochs", epoch));
                break;
            }

            // Prepare the next iteration
            epoch++;
            validationErrorPrev = validationErrorCur;
        }

        // Save the trained network
        String fileName = String.format("%s%dx%d.nn", networkName, networkDepth, hiddenLayerWidth);
        try {
            nn.save(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void loadAndTest() {
        final String networkFileName = "/mnist2x30.nn";
        final String testSetFileName = "/mnist_test_10.csv";

        InputStream stream = Program.class.getResourceAsStream(networkFileName);
        NeuralNetwork nn = new NeuralNetwork(stream);
        float testError = test(nn, testSetFileName, false);
        System.out.println(String.format("Avg. test error: %f", testError));
    }

    private static float train(NeuralNetwork nn, String fileName) {
        int counter = 0;
        float errorTotal = 0.0F;

        InputStream stream = Program.class.getResourceAsStream(fileName);
        Scanner scanner = new Scanner(stream).useDelimiter(",|\\r\\n|\\r|\\n");
        while (scanner.hasNextInt()) {
            int value = scanner.nextInt();

            int[] data = new int[28 * 28];
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    data[i * 28 + j] = scanner.nextInt();
                }
            }

            TrainingInput input = new TrainingInput(new SensorInput(data), value);
            float[] actual = nn.train(input.getScaledInput(), input.getLabel());
            float[] expected = input.getLabel();
            errorTotal += getError(actual, expected);

            counter++;

            // Visualize the process
            if (counter % 100 == 0) {
                System.out.print(".");
            }
            if (counter % 5000 == 0) {
                System.out.println("");
            }
        }

        System.out.println("");
        return errorTotal / counter;
    }

    public static float test(NeuralNetwork nn, String fileName, boolean debug) {
        int counterTotal = 0;
        int counterCorrect = 0;
        float errorTotal = 0.0F;

        InputStream stream = Program.class.getResourceAsStream(fileName);
        Scanner scanner = new Scanner(stream).useDelimiter(",|\\r\\n|\\r|\\n");
        while (scanner.hasNextInt()) {
            int value = scanner.nextInt();

            int[] data = new int[28 * 28];
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    data[i * 28 + j] = scanner.nextInt();
                }
            }

            TrainingInput input = new TrainingInput(new SensorInput(data), value);
            float[] actual = nn.classify(input.getScaledInput());
            float[] expected = input.getLabel();
            errorTotal += getError(actual, expected);

            counterTotal++;
            if (isMatch(actual, expected)) {
                counterCorrect++;
            }

            if (debug) {
                System.out.print("ACTUAL  : ");
                for (int i = 0; i < actual.length; i++) {
                    System.out.print(String.format("%.2f ", actual[i]));
                }
                System.out.print("\nEXPECTED: ");
                for (int i = 0; i < expected.length; i++) {
                    System.out.print(String.format("%.2f ", expected[i]));
                }
                System.out.println();
                System.out.println();
            }
        }

        System.out.println(String.format("Correct: %d out of %d", counterCorrect, counterTotal));
        return errorTotal / counterTotal;
    }

    private static boolean isMatch(float[] actual, float[] expected) {
        int actualValue = 0;
        float actualMax = actual[0];

        int expectedValue = 0;
        float expectedMax = expected[0];

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] > actualMax) {
                actualMax = actual[i];
                actualValue = i;
            }
            if (expected[i] > expectedMax) {
                expectedMax = expected[i];
                expectedValue = i;
            }
        }
        return actualValue == expectedValue;
    }

    private static float getError(float[] actual, float[] expected) {
        float error = 0.0F;
        for (int i = 0; i < actual.length; i++) {
            error += Math.pow(expected[i] - actual[i], 2);
        }
        return error / actual.length;
    }

    private static int getLabel(float[] actual) {
        int actualValue = 0;
        float actualMax = actual[0];

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] > actualMax) {
                actualMax = actual[i];
                actualValue = i;
            }
        }

        return actualValue;
    }
}
