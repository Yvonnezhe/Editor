package editor;


import javafx.scene.text.Text;

/**
 * Created by Yao on 2016/10/1.
 */
public class FastLinkedList {
    private FastLinkedListNode sentinel;
    private int curX;
    private int curY;
    private FastLinkedListNode currentNode;
    private int size;
    private int fontSize = 12;
    private String fontName = "Verdana";

    public FastLinkedList() {
        sentinel = new FastLinkedListNode();
        curX = 0;
        curY = 0;
        currentNode = sentinel;
        size = 0;
    }
    //get methods
    public FastLinkedListNode getSentinel(){return sentinel;}
    public FastLinkedListNode getCurrentNode() {return currentNode;}
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
    public void updateCurrentNode(FastLinkedListNode node) {
        currentNode = node;
        updateCurXY(node.getXPos(), node.getYPos());
    }
    public int getSize(){return  size;}
    //add a node
    public void add(FastLinkedListNode node) {
        if (getCurrentNode().getNext() == null) {
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
        int charWidth = (int) Math.round(node.getItem().getLayoutBounds().getWidth());
        int charHeight = (int) Math.round(node.getItem().getLayoutBounds().getHeight());
    }
    //remove the current node
    public FastLinkedListNode remove() {
        if (getSize() == 0) {return null;}
        FastLinkedListNode ret = currentNode;
        if (getCurrentNode().getNext() == null) {
            currentNode = currentNode.getPrev();
            currentNode.setNext(null);
        } else {
            currentNode = currentNode.getPrev();
            currentNode.getNext().getNext().setPrev(currentNode);
            currentNode.setNext(currentNode.getNext().getNext());
        }
        size--;
        return ret;
    }
    public int findCurLenDiffForUpByXY(int x, int y) {
        if (y > currentNode.getYPos()) {
            return 0;
        }
        FastLinkedListNode ptr = getCurrentNode();
        FastLinkedListNode rst = getCurrentNode();
        int minusLen = 0;
        while (ptr != getSentinel()) {
            if (ptr.getYPos() > y) {
                minusLen += ptr.getCharWidth();
                ptr = ptr.getPrev();
                rst = ptr;
                continue;
            } else {
                if (x <= ptr.getXPos()) {
                    rst = ptr;
                    minusLen += ptr.getCharWidth();
                    ptr = ptr.getPrev();
                } else {
                    break;
                }
            }
        }
        updateCurrentNode(rst);
        return minusLen - ptr.getCharWidth();
    }
    public int findCurLenForDownByXY(int x, int y) {
        if (y < currentNode.getYPos()) {
            return 0;
        }
        FastLinkedListNode ptr = getCurrentNode();
        FastLinkedListNode rst = getCurrentNode();
        int addedLen = 0;
        while (ptr != null && ptr.getNext() != null) {
            if (ptr.getYPos() < y) {
                addedLen += ptr.getCharWidth();
                ptr = ptr.getNext();
                rst = ptr;
                continue;
            } else {
                if (x >= ptr.getXPos()) {
                    rst = ptr;
                    addedLen += ptr.getCharWidth();
                    ptr = ptr.getNext();
                } else {
                    break;
                }
            }
        }
        updateCurrentNode(rst);
        return addedLen - ptr.getCharWidth();
    }
    public int findCurLenForMouse(int x, int y) {
        FastLinkedListNode ptr = getSentinel().getNext();
        FastLinkedListNode newCur = getCurrentNode();
        int sum = 0;
        while (ptr != null) {
            if (ptr.getYPos() < y) {
                sum += ptr.getCharWidth();
                newCur = ptr;
                ptr = ptr.getNext();
            } else if (ptr.getYPos() == y) {
                if (ptr.getXPos() < x) {
                    sum += ptr.getCharWidth();
                    newCur = ptr;
                    ptr = ptr.getNext();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        updateCurrentNode(newCur);
        return sum;
    }
    public static void main(String[] args) {
        FastLinkedList f = new FastLinkedList();
        f.add(new FastLinkedListNode(new Text("a")));

        f.add(new FastLinkedListNode(new Text("b")));
        f.add(new FastLinkedListNode(new Text("c")));
        f.add(new FastLinkedListNode(new Text("d")));
        f.updateCurrentNode(f.getCurrentNode().getPrev());
        f.add(new FastLinkedListNode(new Text("x")));
        System.out.println(f.getSize());
        System.out.println("currentnode: " + f.getCurrentNode().getItem().getText() +" == x");
        FastLinkedListNode ptr1 = f.getSentinel().getNext();
        while(ptr1!=null) {
            System.out.print(ptr1.getItem().getText() + " ");
            ptr1 = ptr1.getNext();
        }
        System.out.println(f.getSize());
        System.out.println();
        f.remove();
        FastLinkedListNode ptr = f.getSentinel().getNext();
        while(ptr!=null) {
            System.out.print(ptr.getItem().getText() + " ");
            ptr = ptr.getNext();
        }
        System.out.println(f.getSize());
        System.out.println();
        System.out.println("currentnode: " + f.getCurrentNode().getItem().getText() +" == c");
    }
}
