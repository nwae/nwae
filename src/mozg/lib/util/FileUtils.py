#!/use/bin/python
# --*-- coding: utf-8 --*--

import ie.lib.util.StringUtils as su


class FileUtils(object):

    def __init__(self):
        return

    @staticmethod
    def read_text_file(filepath, encoding='utf-8'):
        try:
            fh = open(filepath, 'r', encoding=encoding)
        except IOError as e:
            print('Can\'t open file [' + filepath + ']. ' + e.strerror)
            return []

        lines = []
        for line in fh:
            # Can just use StringUtils.trim() to remove newline also
            # if remove_newline:
            #    line = re.sub('\n|\r', '', line)
            # line = unicode(line, encoding)
            lines.append(line)

        fh.close()
        return lines

    @staticmethod
    def read_text_files(filepaths, encoding='utf-8', verbose=False):
        contents = []
        for file in filepaths:
            if verbose:
                print( 'Reading file [' + file + ']...' )
            lines = FileUtils.read_text_file(file, encoding)
            contents = contents + lines

            if verbose:
                print( '   Read ' + str(len(lines)) + ' lines.' )

        return contents

    @staticmethod
    def read_config_file(filepath, pv, encoding='utf-8'):
        server_config = FileUtils.read_text_file(filepath=filepath, encoding=encoding)
        for line in server_config:
            line = su.StringUtils.trim(line)
            line = line.lower()
            line_split = line.split(sep=' ')
            if len(line_split) == 2:
                param = line_split[0]
                value = line_split[1]
                if param in list(pv.keys()):
                    pv[param] = value
                else:
                    errmsg = 'Unknown param [' + param + '] in config file [' + filepath + ']'
                    raise (Exception(errmsg))
        return(pv)


# Demo to read this file itself
def demo_1():
    lines = FileUtils.read_text_file('FileUtils.py')
    for i in range(0, len(lines), 1):
        print(su.StringUtils.remove_newline(lines[i]))

    print(lines)
    return


# demo_1()
