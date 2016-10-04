package editor;

import javafx.scene.input.KeyCode;
import javafx.scene.text.Text;

/**
 * Created by Administrator on 2016/10/1.
 */
public class HistoryEvent {
    private KeyCode code;
    private Text text;
    private int x;
    private int y;
    public HistoryEvent(KeyCode code, Text text, int x, int y){
        this.code = code;
        this.text = text;
        this.x = x;
        this.y = y;
    }
    public int getX(){
        return x;
    }
    public int getY() {
        return y;
    }
    public KeyCode getCode() {
        return code;
    }
    public Text getText() {
        return text;
    }
    public void setX(int x) {
        this.x = x;
    }
    public void setY(int y) {
        this.y = y;
    }
    public void setCode(KeyCode code) {
        this.code = code;
    }
    public void setText(Text text) {
        this.text = text;
    }
}
