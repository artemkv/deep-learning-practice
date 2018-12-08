package net.artemkv.ai.deeplearning;

class Matrix {
    private final int rows;
    private final int columns;
    private final float[][] matrix;

    public Matrix(int rows, int columns) {
        if (rows <= 0) {
            throw new IllegalArgumentException("rows");
        }
        if (columns <= 0) {
            throw new IllegalArgumentException("columns");
        }

        this.rows = rows;
        this.columns = columns;

        matrix = new float[rows][columns];
    }

    public Matrix(int rows, int columns, MatrixInitializer initializer) {
        this(rows, columns);

        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                matrix[row][column] = initializer.getValue(row, column);
            }
        }
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public float getValue(int row, int column) {
        return matrix[row][column];
    }

    public float[] multiplyVector(float[] vector) {
        if (vector.length != columns) {
            throw new IllegalArgumentException(
                String.format("cannot multiple matrix of %d columns to vector of length %d",
                    columns,
                    vector.length)
            );
        }

        float[] result = new float[rows];
        for (int row = 0; row < rows; row++) {
            float val = 0;
            for (int column = 0; column < columns; column++) {
                val += matrix[row][column] * vector[column];
            }
            result[row] = val;
        }
        return result;
    }

    public float[] invertAndMultiplyVector(float[] vector) {
        if (vector.length != rows) {
            throw new IllegalArgumentException(
                String.format("cannot multiple inverted matrix of %d rows to vector of length %d",
                    rows,
                    vector.length)
            );
        }

        float[] result = new float[columns];
        for (int column = 0; column < columns; column++) {
            float val = 0;
            for (int row = 0; row < rows; row++) {
                val += matrix[row][column] * vector[row];
            }
            result[column] = val;
        }
        return result;
    }

    public void applyDelta(Matrix delta) {
        if (delta.getRows() != rows) {
            throw new IllegalArgumentException(
                String.format("cannot add matrix of %d rows to matrix of %d rows",
                    delta.getRows(),
                    rows)
            );
        }
        if (delta.getColumns() != columns) {
            throw new IllegalArgumentException(
                String.format("cannot add matrix of %d columns to matrix of %d columns",
                    delta.getColumns(),
                    columns)
            );
        }

        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                matrix[row][column] += delta.getValue(row, column);
            }
        }
    }

    public String serialize() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                sb.append(String.format("%f;", matrix[row][column]));
            }
        }
        sb.setLength(sb.length() - 1);
        return sb.toString();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                sb.append(matrix[row][column] + ", ");
            }
            sb.setLength(sb.length() - 2);
            sb.append(String.format("%n"));
        }
        return sb.toString();
    }
}
