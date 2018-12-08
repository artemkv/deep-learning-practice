package net.artemkv.ai.deeplearning;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ActivationFunctionsTest {
    @Test
    public void testSigmoid() {
        float y = ActivationFunctions.sigmoid(1.05F);
        assertEquals(0.74077487F, y, "sigmoid of 1.05");
    }
}
