# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.utils.UnitTest as ut
import hashlib
from nwae.utils.Hash import Hash


class Obfuscate:

    STRING_ENCODING = 'utf-8'

    def __init__(
            self,
    ):
        return

    def strdigest(
            self,
            # bytes list
            bytes_list
    ):
        # Convert back to string
        return ''.join(chr(x) for x in bytes_list)

    def hexdigest(
            self,
            # bytes list
            bytes_list,
            # If provided, we convert to the unicode characters
            unicode_range = None
    ):
        # Convert back to string
        str_hex = ''
        for b in bytes_list:
            if b >= 16:
                str_b = str(hex(b))[2:4]
            else:
                str_b = '0' + str(hex(b))[2:3]
            # print(str_b)
            str_hex += str_b

        if unicode_range:
            return Hash.convert_ascii_string_to_other_alphabet(
                ascii_char_string = str_hex,
                unicode_range     = unicode_range,
                group_n_char      = 4
            )
        else:
            return '0x' + str_hex

    def hash_compression(
            self,
            s,
            # By default we return the original hash
            desired_byte_length = 32
    ):
        if desired_byte_length % 4 != 0:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Desired byte length must be 0 modulo-4, given = ' + str(desired_byte_length)
            )

        m = hashlib.sha256()
        m.update(bytes(s, encoding=Obfuscate.STRING_ENCODING))
        # This will return a bytes list of length 32
        h = m.digest()
        if len(h) % 4 != 0:
            raise Exception(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Hash bytes length must be 0 modulo-4, got = ' + str(h)
            )

        # We compress to 8 bytes from the 32 bytes
        # The original SHA-256 appends 8 parts concatenated together, we break into 4 parts and xor them

        # 4 blocks
        n_blocks = int( len(h) / desired_byte_length )
        # 8 bytes block length
        block_len = int( len(h) / n_blocks )
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Number of blocks = ' + str(n_blocks) + ', block length = ' + str(block_len)
        )

        # First block
        bytes_xor = h[0:block_len]
        for i in range(1, n_blocks, 1):
            idx_start = i * block_len
            idx_end = (i+1) * block_len
            cur_block = h[idx_start:idx_end]
            if len(bytes_xor) != len(cur_block):
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Different block lengths "' + str(bytes_xor)
                    + '", and "' + str(cur_block) + '"'
                )
            bytes_xor = self.xor_bytes(
                b1 = bytes_xor,
                b2 = cur_block
            )

        return bytes_xor

    def xor_bytes(
            self,
            b1,
            b2
    ):
        t12 = zip(b1,b2)

        res_xor = []
        for x in t12:
            byte_xor = x[0] ^ x[1]
            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + 'XOR "' + str(hex(x[0])) + '" and "' + str(hex(x[1])) + '" = ' + str(hex(byte_xor))
            )
            res_xor.append(byte_xor)

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': XOR between "' + str(self.hexdigest(b1))
            + '" and "' + str(self.hexdigest(b2))
            + '" = "' + str(self.hexdigest(res_xor)) + '"'
        )

        return res_xor

    #
    # Given 2 strings, we take the XOR of the bytes
    # We append to the shorter string in a repeat manner so that string lengths are equal
    #
    def xor_string(
            self,
            s1,
            s2
    ):
        Log.debug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': XOR between "' + str(s1) + '" and "' + str(s2) + '".'
        )

        len_s1 = len(s1)
        len_s2 = len(s2)
        len_max = max(len(s1), len(s2))

        # Append to the shorter one, in a repeat manner
        for i in range(len(s1), len_max, 1):
            s1 += s1[(i-len_s1)]
        for i in range(len(s2), len_max, 1):
            s2 += s2[(i-len_s2)]

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After appending, XOR between "' + str(s1) + '" and "' + str(s2) + '".'
        )

        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': s1 "' + str(s1) + '", s2 "' + str(s2) + '"'
        )

        b1 = bytes(s1, encoding=Obfuscate.STRING_ENCODING)
        b2 = bytes(s2, encoding=Obfuscate.STRING_ENCODING)

        bytes_xor = self.xor_bytes(
            b1 = b1,
            b2 = b2
        )

        return bytes_xor


class ObfuscateUnitTest:

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        self.obfuscater = Obfuscate()
        return

    def run_unit_test(
            self
    ):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        test_cases = [
            ['abcde', '12345678',
             [0x50,0x50,0x50,0x50,0x50,0x57,0x55,0x5b], 'PPPPPWU['],
        ]
        for x in test_cases:
            s1 = x[0]
            s2 = x[1]
            expected_bytes = x[2]
            expected_string = x[3]

            observed_bytes = self.obfuscater.xor_string(s1=s1, s2=s2)
            observed_string = self.obfuscater.strdigest(bytes_list=observed_bytes)

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_bytes,
                expected = expected_bytes,
                test_comment = 'Test "' + str(s1) + '", "' + str(s2) + '" = ' + str(expected_bytes)
            ))
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_string,
                expected = expected_string,
                test_comment = 'Test "' + str(s1) + '", "' + str(s2) + '" = ' + str(expected_string)
            ))

        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    ObfuscateUnitTest(ut_params=None).run_unit_test()

    obj = Obfuscate()
    hc = obj.hash_compression(
        s = '',
        desired_byte_length = 16
    )
    print('Got length = ' + str(len(hc)))

    print(obj.hexdigest(
        bytes_list = hc,
        unicode_range = Hash.BLOCK_CHINESE
    ))

    exit(0)
