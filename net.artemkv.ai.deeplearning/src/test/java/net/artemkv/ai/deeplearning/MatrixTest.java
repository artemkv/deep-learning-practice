package net.artemkv.ai.deeplearning;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MatrixTest {
    @Test
    public void testCreation() {
        Matrix matrix = new Matrix(5, 7);

        assertEquals(5, matrix.getRows(), "rows");
        assertEquals(7, matrix.getColumns(), "columns");
        assertEquals(0.0F, matrix.getValue(0, 0), "value");
    }

    @Test
    public void testInitialize() {
        Matrix matrix = new Matrix(3, 3, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return column * 10 + row;
            }
        });

        assertEquals(21.0F, matrix.getValue(1, 2), "value");
    }

    @Test
    public void testMultipleSquareMatrixToVector() {
        final AtomicInteger i = new AtomicInteger(1);
        Matrix matrix = new Matrix(2, 2, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return i.incrementAndGet();
            }
        });

        float[] output = matrix.multiplyVector(new float[] { 6, 7 });
        assertEquals(2.0F * 6.0F + 3.0F * 7.0F, output[0], "value 0");
        assertEquals(4.0F * 6.0F + 5.0F * 7.0F, output[1], "value 1");
    }

    @Test
    public void testMultipleHorisontalMatrixToVector() {
        final AtomicInteger i = new AtomicInteger(1);
        Matrix matrix = new Matrix(2, 3, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return i.incrementAndGet();
            }
        });

        float[] output = matrix.multiplyVector(new float[] { 6, 7, 8 });
        assertEquals(2.0F * 6.0F + 3.0F * 7.0F + 4.0F * 8.0F, output[0], "value 0");
        assertEquals(5.0F * 6.0F + 6.0F * 7.0F + 7.0F * 8.0F, output[1], "value 1");
    }

    @Test
    public void testMultipleVerticalMatrixToVector() {
        final AtomicInteger i = new AtomicInteger(1);
        Matrix matrix = new Matrix(3, 2, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return i.incrementAndGet();
            }
        });

        float[] output = matrix.multiplyVector(new float[] { 6, 7 });
        assertEquals(2.0F * 6.0F + 3.0F * 7.0F, output[0], "value 0");
        assertEquals(4.0F * 6.0F + 5.0F * 7.0F, output[1], "value 1");
        assertEquals(6.0F * 6.0F + 7.0F * 7.0F, output[2], "value 2");
    }

    @Test
    public void testInvertAndMultipleToVector() {
        final AtomicInteger i = new AtomicInteger(1);
        Matrix matrix = new Matrix(2, 2, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return i.incrementAndGet();
            }
        });

        float[] output = matrix.invertAndMultiplyVector(new float[] { 6, 7 });
        assertEquals(2.0F * 6.0F + 4.0F * 7.0F, output[0], "value 0");
        assertEquals(3.0F * 6.0F + 5.0F * 7.0F, output[1], "value 1");
    }

    @Test
    public void testApplyDelta() {
        final AtomicInteger i = new AtomicInteger(1);
        Matrix matrix = new Matrix(2, 2, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return i.incrementAndGet();
            }
        });
        Matrix delta = new Matrix(2, 2, new MatrixInitializer() {
            public float getValue(int row, int column) {
                return 0.1F;
            }
        });

        matrix.applyDelta(delta);
        assertEquals(2.1F, matrix.getValue(0, 0), "value 0,0");
        assertEquals(3.1F, matrix.getValue(0, 1), "value 0,1");
        assertEquals(4.1F, matrix.getValue(1, 0), "value 1,0");
        assertEquals(5.1F, matrix.getValue(1, 1), "value 1,1");
    }
}