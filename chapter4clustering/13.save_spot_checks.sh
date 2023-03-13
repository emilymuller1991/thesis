INPUT=outputs/spots/walthamstow_matched_centroid.txt
OLDIFS=$IFS
IFS=' '
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
sed 2d $rpt | while read c1 c2 i1 i2
do
    echo $i1, $i2
	cp "/media/emily/south/emily_phd_images/census_2011/${i1}.png" /media/emily/south/emily_phd_images/walthamstow_matched_centroid/
    cp "/media/emily/south/emily_phd_images/census_2021/${i2}.png" /media/emily/south/emily_phd_images/walthamstow_matched_centroid/
done < $INPUT
IFS=$OLDIFS