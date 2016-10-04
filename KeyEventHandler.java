package editor;

import javafx.event.EventHandler;
import javafx.scene.input.KeyEvent;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.VPos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.util.Duration;
/**
 * Created by Yao on 2016/10/1.
 */

public class KeyEventHandler implements EventHandler<KeyEvent> {
    int textCenterX;
    int textCenterY;

    private static final int STARTING_TEXT_POSITION_X = 250;
    private static final int STARTING_TEXT_POSITION_Y = 250;
    private static final int STARTING_FONT_SIZE = 20;

    /**
     * The Text to display on the screen.
     */
    private Text displayText = new Text(STARTING_TEXT_POSITION_X, STARTING_TEXT_POSITION_Y, "");
    private int fontSize = STARTING_FONT_SIZE;

    private String fontName = "Verdana";

    //10.2
    public Editor editor;
    Group root;
    KeyEventHandler(final Group root, int windowWidth, int windowHeight, Editor editor) {
        this.root = root;
        this.editor = editor;
        textCenterX = 15;
        textCenterY = 0;
//        FastLinkedListNode temp = editor.buffer.getCurrentNode();
//        //System.out.println(temp.getItem());
//        while (temp != null) {
//            Text displayText = temp.getItem();
//            displayText.setFont(Font.font(fontName, fontSize));
//            displayText.setX(editor.positionX);
//            displayText.setY(editor.positionY);
//            //displayText.setText(displayText.getText()+temp.getItem().getText());
//            root.getChildren().add(displayText);
//            int charWidth = (int) Math.round(displayText.getLayoutBounds().getWidth());
//            int charHeight = (int) Math.round(displayText.getLayoutBounds().getHeight());
//            editor.positionX += charWidth;
//            //editor.positionY += charHeight;
//            temp = temp.getNext();
//        }

        // Initialize some empty text and add it to root so that it will be displayed.
        //displayText = new Text(textCenterX, textCenterY, "");
        // Always set the text origin to be VPos.TOP! Setting the origin to be VPos.TOP means
        // that when the text is assigned a y-position, that position corresponds to the
        // highest position across all letters (for example, the top of a letter like "I", as
        // opposed to the top of a letter like "e"), which makes calculating positions much
        // simpler!
        //displayText.setTextOrigin(VPos.TOP);
        //displayText.setFont(Font.font(fontName, fontSize));
        //root.getChildren().add(displayText);
    }

    @Override
    public void handle(KeyEvent keyEvent) {
        if (keyEvent.getEventType() == KeyEvent.KEY_TYPED) {
            // Use the KEY_TYPED event rather than KEY_PRESSED for letter keys, because with
            // the KEY_TYPED event, javafx handles the "Shift" key and associated
            // capitalization.
            String characterTyped = keyEvent.getCharacter();
            if (characterTyped.length() > 0 && characterTyped.charAt(0) != 8) {
                // Ignore control keys, which have zero length, as well as the backspace
                // key, which is represented as a character of value = 8 on Windows.
                //displayText.setText(characterTyped);

                editor.add(characterTyped);
                FastLinkedListNode temp = editor.buffer.getCurrentNode();
                //System.out.println(temp.getItem());
                while (temp != null) {
                    Text displayText = temp.getItem();
                    displayText.setFont(Font.font(fontName, fontSize));
                    displayText.setX(editor.positionX);
                    displayText.setY(editor.positionY);
            //displayText.setText(displayText.getText()+temp.getItem().getText());
                     root.getChildren().add(displayText);

                    int charWidth = (int) Math.round(displayText.getLayoutBounds().getWidth());
                    int charHeight = (int) Math.round(displayText.getLayoutBounds().getHeight());
                    editor.positionX += charWidth;
                    //editor.positionY += charHeight;
                    temp = temp.getNext();
                }

                keyEvent.consume();
            }
            else if(characterTyped.charAt(0) == 8)
                editor.delete();


        } else if (keyEvent.getEventType() == KeyEvent.KEY_PRESSED) {
            // Arrow keys should be processed using the KEY_PRESSED event, because KEY_PRESSED
            // events have a code that we can check (KEY_TYPED events don't have an associated
            // KeyCode).
            KeyCode code = keyEvent.getCode();
            if (code == KeyCode.UP) {
                fontSize += 5;
                displayText.setFont(Font.font(fontName, fontSize));
            } else if (code == KeyCode.DOWN) {
                fontSize = Math.max(0, fontSize - 5);
                displayText.setFont(Font.font(fontName, fontSize));
            }
        }
    }
}
