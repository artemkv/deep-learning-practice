package net.artemkv.ai.deeplearning.training;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class ImageLoader {
    public static int[] getData(InputStream stream) throws IOException {
        BufferedImage image = ImageIO.read(stream);

        int width = image.getWidth();
        int height = image.getHeight();

        int[] data = new int[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);

                int red = (rgb >> 16) & 0x000000FF;
                int green = (rgb >> 8) & 0x000000FF;
                int blue = (rgb) & 0x000000FF;

                data[y * width + x] = 254 - (int) (0.2126 * red + 0.7152 * green + 0.0722 * blue);
            }
        }

        return data;
    }
}
