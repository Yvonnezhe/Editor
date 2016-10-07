package editor;

import javafx.application.Application;
import javafx.beans.value.ChangeListener;
import javafx.event.EventHandler;
import javafx.geometry.VPos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.control.ScrollBar;
import javafx.stage.Stage;
import javafx.geometry.Orientation;
import javafx.beans.value.ObservableValue;
import javafx.scene.shape.Rectangle;
import javafx.scene.input.MouseEvent;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
/**
 * Created by Yao on 2016/10/1.
 */
public class Editor extends Application{
    private static String fileNme = "Tan and Yao";
    private Scene scene;
    private int windowWidth = 500;
    private int windowHeight = 500;
    public Cursor cur = new Cursor();;
    private int line_length;
    protected FastLinkedList buffer = new FastLinkedList();
    public int positionX = 0;
    public int positionY = 100;
    public int textLength = 0;// the total length
    public int curLength = 0;//from head to the cursor
    //private Text displayText = new Text(STARTING_TEXT_POSITION_X, STARTING_TEXT_POSITION_Y, "");
    private String fontName = "Verdana";
    private int fontSize = 20;
    private History history;
    boolean startEvent = false;
    public void delete () {
        FastLinkedListNode temp = buffer.remove();


    }
    public void add (String character) {
        //startEvent = true;

        Text current  = new Text(character);
        FastLinkedListNode temp = new FastLinkedListNode(current);
        buffer.add(temp);
        //may change after adding cursor
        //int eventX = positionX + (int)Math.round(current.getLayoutBounds().getWidth());
        //int eventY = positionY + (int)Math.round(current.getLayoutBounds().getHeight());
        //HistoryEvent event = new HistoryEvent(KeyCode.ADD, current, eventX, eventY );
        //history.newEvent(event);
    }
    public void handle () {
        return;
    }


    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle(fileNme);
        Group root = new Group();
        scene = new Scene(root, windowWidth, windowHeight, Color.WHITE);
        //add a scrollbar
        ScrollBar scrollBar = new ScrollBar();
        scrollBar.setOrientation(Orientation.VERTICAL);
        scrollBar.setPrefHeight(windowHeight);
        scrollBar.setMin(10);
        scrollBar.setMax(windowHeight);
        scrollBar.setLayoutX(windowWidth - scrollBar.getLayoutBounds().getWidth() - 8);
        root.getChildren().add(scrollBar);


        root.getChildren().add(cur.textBox);
        cur.makeColorChange();

        //bring in keyEventHandler
        EventHandler<KeyEvent> keyEventHandler = new KeyEventHandler(root, this);
        EventHandler<MouseEvent>  mouseEventHandler = new MouseEventHandler(root, this);
        scene.setOnKeyTyped(keyEventHandler);
        scene.setOnKeyPressed(keyEventHandler);
        scene.setOnMouseClicked(mouseEventHandler);
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    public static void main (String[] args){
        launch(args);
    }
}
