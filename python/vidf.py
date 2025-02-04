def d2b(n,r):               #here r is no of bits
  binary = ''
  if n == 0:
    while len(binary) < r:
        binary = '0' + binary
  while n > 0:
    binary = str(n % 2) + binary
    n //= 2
  while len(binary) < r:
    binary = '0' + binary

  return binary


def b2d(binary_string):
  decimal = 0
  for i in range(len(binary_string)):
    digit = int(binary_string[i])
    decimal += digit * 2 ** (len(binary_string) - 1 - i)
  return decimal


def vadf_enc(num):               #input must be +ve int value
  if(num <=1):
    return("0000000000000000")
  else:
    bnum = d2b(num,32)
    for i in range(len(bnum)):
        if bnum[i] == '1':
            loc = 31 - i
            break
    bloc = d2b(loc,5)
    if (loc <6):
        data_bits = bnum[-loc:]
        while len(data_bits) < 6:
            data_bits = '0' + data_bits
    elif (loc ==6):
        data_bits = bnum[32 - loc: 38 - loc]
    else:
        data_bits = bnum[32 - loc: 37 - loc]
        if(bnum[38-loc]=='1'):
           data_bits = data_bits + '1'
        else:
           data_bits = data_bits + bnum[37-loc]

    
    count_ones = data_bits.count('1')
    if (count_ones % 2 == 0):
       parity_bit = '0'
    else:
       parity_bit = '1'

    int_loc = [int(bit) for bit in bloc[::-1]]
    error_correction_bits = [0] * 4
    error_correction_bits[3] = int_loc[4] ^ int_loc[2] ^ int_loc[1]
    error_correction_bits[2] = int_loc[4] ^ int_loc[3] ^ int_loc[1]
    error_correction_bits[1] = int_loc[4] ^ int_loc[3] ^ int_loc[2]
    error_correction_bits[0] = int_loc[0]
    
    temp = ''.join(map(str, error_correction_bits))
    ecb = temp[::-1]

    out16 = parity_bit + bloc + ecb + data_bits
    return out16

def hamming_code_decode(instream):
  check = [int(bit) for bit in instream[0:10]]
  s = [0] * 4
  s[0] = check[6] ^ check[1] ^ check[3] ^ check[4]
  s[1] = check[7] ^ check[1] ^ check[2] ^ check[4]
  s[2] = check[8] ^ check[1] ^ check[2] ^ check[3]
  s[3] = check[9] ^ check[5]
  syn = ''.join(map(str, s))
  inp = [int(bit) for bit in instream]
  if (syn[3] == "1"):
    inp[5] = 1 ^ inp[5]
  if (syn[:-1] == "001"):
    inp[8] = 1 ^ inp[8]
  if (syn[:-1] == "010"):
    inp[7] = 1 ^ inp[7]
  if (syn[:-1] == "100"):
    inp[6] = 1 ^ inp[6]
  if (syn[:-1] == "011"):
    inp[2] = 1 ^ inp[2]
  if (syn[:-1] == "101"):
    inp[3] = 1 ^ inp[3]
  if (syn[:-1] == "110"):
    inp[4] = 1 ^ inp[4]
  if (syn[:-1] == "111"):
    inp[1] = 1 ^ inp[1]
  return ''.join(map(str, inp))


def vadf_dec_ideal(inp):
    instream = hamming_code_decode(inp)
    if(instream == "0000000000000000" ):
      return 0
    outstream = ""
    loc = 31 - b2d(instream[1:6])
    for i in range(loc):
       outstream = outstream + '0'
    outstream = outstream + '1'
    if ((31 - loc) < 6 ):
      k = 31 - loc
    else:
      k = 6
    for i in range(k):
        outstream = outstream + instream[i-k]
    while len(outstream) < 32:
        outstream = outstream + '0'
    return b2d(outstream)


