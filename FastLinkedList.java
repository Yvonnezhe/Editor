package editor;


import javafx.scene.text.Text;

/**
 * Created by Yao on 2016/10/1.
 */
public class FastLinkedList {
    private FastLinkedListNode sentinel;
    private int currentIndex;
    private int curX;
    private int curY;
    private FastLinkedListNode currentNode;
    private int size;
    private int fontSize = 12;
    private String fontName = "Verdana";

    public FastLinkedList() {
        sentinel = new FastLinkedListNode(new Text(5, 0, ""), null, null);
        currentIndex = 0;
        curX = 5;
        curY = 100;
        currentNode = sentinel;
        size = 0;
    }
    //get methods
    public FastLinkedListNode getSentinel(){return sentinel;}
    public FastLinkedListNode getCurrentNode() {return currentNode;}
    public int getCurrentIndex() {return currentIndex;}
    public int getCurX() {
        return curX;
    }
    public int getCurY() {
        return curY;
    }
    // update current x,y
    public void updateCurXY(int curX, int curY) {
        this.curX = curX;
        this.curY = curY;
    }
    //add a node
    public void add(FastLinkedListNode node) {
        if (currentIndex == size) {
            currentNode.setNext(node);
            node.setPrev(currentNode);
            node.setNext(null);
            currentNode = node;
        } else {
            node.setNext(currentNode.getNext());
            currentNode.getNext().setPrev(node);
            currentNode.setNext(node);
            node.setPrev(currentNode);
            currentNode = node;
        }
        size++;
        currentIndex++;
        int charWidth = (int) Math.round(node.getItem().getLayoutBounds().getWidth());
        int charHeight = (int) Math.round(node.getItem().getLayoutBounds().getHeight());

        ///////how to update cur x, y? need to be improved
        updateCurXY(curX + charWidth, curY);
    }
    //remove the current node
    public FastLinkedListNode remove() {
        FastLinkedListNode ret = currentNode;
        int charWidth = (int) Math.round(currentNode.getItem().getLayoutBounds().getWidth());
        int charHeight = (int) Math.round(currentNode.getItem().getLayoutBounds().getHeight());
        if (currentIndex == size) {
            currentNode = currentNode.getPrev();
            currentNode.setNext(null);
        } else {
            currentNode.getNext().setPrev(currentNode.getPrev());
            currentNode.getPrev().setNext(currentNode.getNext());
            currentNode = currentNode.getPrev();
        }
        size--;
        currentIndex--;
        ///////how to update cur x, y?? need to be improved
        updateCurXY(curX - charWidth, curY);
        return ret;
    }
    public static void main(String[] args) {
        FastLinkedList f = new FastLinkedList();
        f.add(new FastLinkedListNode(new Text("first node")));
        System.out.println(f.getCurrentNode().getItem().getText() + " " + f.getCurX() + " " + f.getCurY());
        System.out.println(f.size);
        f.add(new FastLinkedListNode(new Text("second node")));
        System.out.println(f.getCurrentNode().getItem().getText() + " " + f.getCurX() + " " + f.getCurY());
        System.out.println(f.size);
        f.remove();
        System.out.println(f.getCurrentNode().getItem().getText() + " " + f.getCurX() + " " + f.getCurY());
        System.out.println(f.size);
    }
}
