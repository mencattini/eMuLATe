import java.util.*;
import machine.learning.ARL;

public class Programme {

    public static void main(String[] args){

        ArrayList array = new ArrayList<Double>();

        Random r = new Random();

        for(int i=0; i < 15; i++){
            array.add(r.nextDouble());
        }
        ARL arl = new ARL(array, 4);
        arl.trainingLoop(1);

        System.out.println(arl.toString());
    }
}
