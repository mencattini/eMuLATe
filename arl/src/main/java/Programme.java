import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.util.*;
import machine.learning.ARL;

public class Programme {

    //TODO fixe the exemple in java
    public static void main(String[] args) throws FileNotFoundException {

        ArrayList array = new ArrayList<Double>();
        Scanner lineReader = new Scanner(new File("data/EURUSD.dat"));

        while(lineReader.hasNext()){
            String line = lineReader.nextLine();
            String[] tmp = line.split("/")[0].split(" ");
            array.add(Double.parseDouble(tmp[tmp.length - 1]));
        }

        ARL arl = new ARL(13);
        // arl.loop(array.subList(200000,260000), 1000);
        // arl.loop(array.subList(260000,270000), 1000);

        System.out.println(arl.toString());
    }
}
