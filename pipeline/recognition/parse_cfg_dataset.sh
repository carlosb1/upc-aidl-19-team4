INPUT_DIRECTORY="..\/Data"
OUTPUT_DIRECTORY="\/home\/carlosb\/python-workspace\/upc-aidl-19-team4\/datasets\/cfp-dataset\/Data"
sed s/$INPUT_DIRECTORY/$OUTPUT_DIRECTORY/g $1 > $1".new"
sed s/$INPUT_DIRECTORY/$OUTPUT_DIRECTORY/g $2 > $2".new"
