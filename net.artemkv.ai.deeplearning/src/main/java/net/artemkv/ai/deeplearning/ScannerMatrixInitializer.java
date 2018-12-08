package net.artemkv.ai.deeplearning;

import java.util.Scanner;

public class ScannerMatrixInitializer implements MatrixInitializer {
    private final Scanner scanner;

    public ScannerMatrixInitializer(Scanner scanner) {
        if (scanner == null) {
            throw new IllegalArgumentException("scanner");
        }
        this.scanner = scanner;
    }

    @Override
    public float getValue(int row, int column) {
        return scanner.nextFloat();
    }
}
