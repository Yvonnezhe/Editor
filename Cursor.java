package editor;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.shape.*;
import javafx.scene.shape.Rectangle;
import javafx.scene.paint.Color;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.util.Duration;
/**
 * The class of the cursor
 * a rectangle that change color
 * including the position of the cursor now (x, y)
 * could get and set X and Y of cursor
 *
 *
 * Cursor.setX() and Cursor.setY() to change the position
 * Cursor.makeColorChange() to change the color
 */

public class Cursor {
    //make the box wide small and change the color between black and white to make it like a cursor
    Rectangle textBox;
    //curX and curY is the top left corner of the box
    //which denotes the position of the cursor
    //curX is the right side of the current character

    public Cursor(){
        textBox = new Rectangle(1, 0);

    }

    private class RectangleBlinkEventHandler implements EventHandler<ActionEvent> {

        private int currentColorIndex = 0;
        //change between black and white would like a cursor
        private Color[] boxColors =
                {Color.BLACK, Color.WHITE};

        RectangleBlinkEventHandler() {
            // Set the color to be the first color in the list.
            changeColor();
        }

        private void changeColor() {
            textBox.setFill(boxColors[currentColorIndex]);
            currentColorIndex = (currentColorIndex + 1) % boxColors.length;
        }

        @Override
        public void handle(ActionEvent event) {
            changeColor();
        }
    }

    //Makes the text bounding box change color periodically.
    public void makeColorChange() {
        // Create a Timeline that will call the "handle" function of RectangleBlinkEventHandler
        // every 1 second.
        final Timeline timeline = new Timeline();
        // The rectangle should continue blinking forever.
        timeline.setCycleCount(Timeline.INDEFINITE);
        RectangleBlinkEventHandler cursorColorChange = new RectangleBlinkEventHandler();
        KeyFrame keyFrame = new KeyFrame(Duration.seconds(1), cursorColorChange);
        timeline.getKeyFrames().add(keyFrame);
        timeline.play();
    }
}

