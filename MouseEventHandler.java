package editor;

/**
 * Created by Administrator on 2016/10/1.
 */
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;
import javafx.event.EventType;

public class MouseEventHandler implements EventHandler<MouseEvent> {
    Editor editor;
    Group root;
    static int lineLength = 450;
    double mousePressedX;
    double mousePressedY;
    MouseEventHandler(final Group root, Editor editor) {
        this.root = root;
        this.editor = editor;

    }
    @Override
    public void handle(MouseEvent mouseEvent) {
        EventType eventType = mouseEvent.getEventType();
        if (eventType == MouseEvent.MOUSE_CLICKED) {
            double mousePressedX = mouseEvent.getX();
            double mousePressedY = mouseEvent.getY();
            System.out.print(mousePressedX+" ");
            System.out.print(mousePressedY);
            Text displayText = new Text("abcdefg");
            int charHeight = (int) Math.round(displayText.getLayoutBounds().getHeight());
            editor.curLength = editor.buffer.findCurLenForMouse((int)mousePressedX, (int)mousePressedY);
System.out.print(" "+editor.curLength+"curLength");
            editor.cur.setx((editor.curLength) % lineLength);
            editor.cur.sety(charHeight * ((editor.curLength) / lineLength) + 100 - charHeight * 2 / 3);
            System.out.println("  "+editor.cur.getX()+"cursor"+editor.cur.getY());
        }
        else if (eventType == MouseEvent.MOUSE_RELEASED) {

        }

    }
}