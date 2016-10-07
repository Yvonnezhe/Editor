package editor;

import javafx.scene.text.Text;

/**
 * Created by Yao on 2016/10/3.
 */
public class FastLinkedListNode {
    private Text item;
    private int charWidth;
    private FastLinkedListNode prev;
    private FastLinkedListNode next;

    public FastLinkedListNode() {
        item = null;
        prev = null;
        next = null;
        charWidth = 0;
    }

    public FastLinkedListNode(Text text) {item = text;}
    public FastLinkedListNode getNext() {return next;}
    public FastLinkedListNode getPrev() {return prev;}
    public Text getItem() {return item;}
    public int getXPos() {
        return (int) Math.round(item.getX());
    }
    public int getYPos() {return (int) Math.round(item.getY());}
    public int getCharWidth() {return charWidth;}

    public void setPrev(FastLinkedListNode prev) {this.prev = prev;}
    public void setNext(FastLinkedListNode next) {this.next = next;}
    public void setItem(Text item) {this.item = item;}
    public void updateXPos(int newX) {item.setX(newX);}
    public void updateYPos(int newY) {item.setY(newY);}
    public void setCharWidth(int charWidth) {this.charWidth = charWidth;}
}