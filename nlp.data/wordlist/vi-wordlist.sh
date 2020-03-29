grep "js-file-line\">" vn-wordlist.txt | sed s/".*js-file-line\">"//g | sed s/"<.*"//g > tmp
