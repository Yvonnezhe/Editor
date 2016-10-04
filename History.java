package editor;

import java.util.Stack;

/**
 * Created by Administrator on 2016/10/1.
 */
public class History {
    Stack<HistoryEvent> undo;
    Stack<HistoryEvent> redo;

    //need to push to Stack undo
    //and clear Stack redo
    public void newEvent(HistoryEvent event){
        undo.push(event);
        redo.clear();
    }
    //ctrl+Z
    //pop from Stack undo
    //push to Stack redo
    public HistoryEvent operateUndo(){
        HistoryEvent event  = undo.pop();
        redo.push(event);
        return event;
    }
    //ctrl+Y
    //pop from redo
    public HistoryEvent operateRedo(){
        HistoryEvent event = redo.pop();
        return event;
    }
}



