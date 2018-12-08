package net.artemkv.ai.deeplearning.training;

import net.artemkv.ai.deeplearning.NeuralNetwork;

import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

public class Program {
    public static void main(String[] args) {
        //trainAndSave();
        //loadAndTest();
    }

    private static void trainAndSave() {
        NeuralNetwork nn = new NeuralNetwork(2, 28 * 28, 100, 10);
        train(nn, "/mnist_train_100.csv");
        try {
            nn.save("mnist_100.nn");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void loadAndTest() {
        InputStream stream = Program.class.getResourceAsStream("/mnist_100.nn");
        NeuralNetwork nn = new NeuralNetwork(stream);
        test(nn, "/mnist_test_10.csv");
    }

    private static void train(NeuralNetwork nn, String fileName) {
        int counter = 0;
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
            nn.train(input.getScaledInput(), input.getLabel());
            counter++;
            if (counter % 1000 == 0) {
                System.out.print(".");
            }
            if (counter % 50000 == 0) {
                System.out.println("");
            }
        }
    }

    public static void test(NeuralNetwork nn, String fileName) {
        int counterTotal = 0;
        int counterCorrect = 0;

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

            counterTotal++;
            if (isMatch(actual, expected)) {
                counterCorrect++;
            }

/*            System.out.print("ACTUAL  : ");
            for (int i = 0; i < actual.length; i++) {
                System.out.print(String.format("%.2f ", actual[i]));
            }
            System.out.print("\nEXPECTED: ");
            for (int i = 0; i < expected.length; i++) {
                System.out.print(String.format("%.2f ", expected[i]));
            }
            System.out.println();
            System.out.println();*/
        }

        System.out.print(String.format("Correct: %d out of %d", counterCorrect, counterTotal));
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

    public static void test() {
        float[] inputA = {
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        };
        float[] inputB = {
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        };
        float[] inputA1 = {
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        };

        NeuralNetwork nn = new NeuralNetwork(3, 10 * 10, 5, 2);

        // Untrained network
        float[] resultBeforeTrainingA = nn.classify(inputA);
        float[] resultBeforeTrainingB = nn.classify(inputB);

        // Train the network
        for (int i = 0; i < 1000; i++) {
            nn.train(inputA, new float[]{1.0F, 0.0F});
            nn.train(inputB, new float[]{0.0F, 1.0F});
        }

        // Verify the trained network
        float[] resultAfterTrainingA = nn.classify(inputA);
        float[] resultAfterTrainingB = nn.classify(inputB);

        // Predict
        float[] resultAfterTrainingA1 = nn.classify(inputA1);

        System.out.println(resultAfterTrainingA1[0] + ", " + resultAfterTrainingA1[1]);
    }
}
