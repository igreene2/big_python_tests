import struct

def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]



c = float_to_bin(-0.0003956189264716169)
print(c)

inte = int(c)
print(inte)
hexy = hex(inte)
print(hexy)

bin =  10111001110011110110101100010011

integer = int(str(bin), 2)
print(integer)
hexa = hex(integer)
print(hexa)

print(bin_to_float(str(inte)))

tester = float.fromhex(str(hexy))
print(tester)


test = 0x7f9e681779de9f67c8494b9a1b
#print(struct.unpack(">f",struct.pack(">i",int('7f9e6817',16)))[0])
[-0.0003956189264716169, -0.0016877538992962208, -0.00020637519205832995, 
-0.000305003595844522, -0.000446614207896013, -0.0005184036468095002,
 0.0003486545758929018, 5.9533502862536416e-05, 0.0011955305165319272, 
 0.001373875795667045, 7.066164032176487e-05, 3.0949557838223745e-05, 
 0.0002533883942390817, 0.0001770020063320765, 5.0173478690645454e-05]