file(REMOVE_RECURSE
  "BackgroundImage.pdb"
  "BackgroundImage.app/Contents/MacOS/BackgroundImage"
)

# Per-language clean rules from dependency scanning.
foreach(lang)
  include(CMakeFiles/BackgroundImage.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
