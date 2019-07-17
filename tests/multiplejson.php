<?php
$allData = array();
$count = 0;
if ($handle = readdir()('./json/')) {
    echo json_encode($handle);
    /*while (false !== ($entry = readdir($handle))) {
        if ($entry != "." && $entry != "..") {

            echo $entry."<br />";

            $source_file = file_get_contents('./json/'.$entry);

            $data = json_decode($source_file,TRUE);

            if($data != null){


                $allData = array_merge($allData,$data);
                echo "<br /><br /><br /><br /> !!!! <br /><br /><br /><br />";
                print_r($allData);

            }
            else{
                echo "Invalid Json File";
            } //end else            
        }
closedir($handle);
}*/
}
//echo "<br /><br /><br /><br /> !!!! <br /><br /><br /><br />";

print_r($allData);  