module vadf #(parameter a = 1,parameter b = 2,parameter c = 3)(
    input clk,rst,
    input [1:0]mode_sel,
    input [31:0] in_data,
    output reg [15:0] out_data_16,
    output reg [11:0] out_data_12,
    output reg [7:0] out_data_8,
    output reg [4:0] location_bits
);
    reg [5:0] data_bits_16;
    reg [3:0] error_correction_bits;
    reg [2:0] data_bits_12;
    reg [1:0] data_bits_8;
    reg parity_bit;
    integer i;

    reg [31:0] _in_data;

    always@(posedge clk) begin
        if(rst)begin
            out_data_16=16'd0;
            out_data_12=12'd0;
            out_data_8=8'd0;
            location_bits=5'b00000;
            data_bits_16=6'b000000;
            error_correction_bits=4'b0000;
            data_bits_12=4'b0000;
            data_bits_8=2'b00;
            parity_bit=1'b0;


        end

        else if((in_data == 32'd0)||(in_data == 32'd1))
            begin
                out_data_16=16'd0;
                out_data_12=12'd0;
                out_data_8=8'd0;
            end

        else begin
            _in_data = in_data;
            for (location_bits = 31; ((_in_data[31] == 0) && (location_bits >0)); location_bits = location_bits - 1) begin
                _in_data = _in_data<<1;
            end


            case(mode_sel)

                // case_1:32 bit to 16 bit approximation
                a: begin
                    out_data_12=12'd0;
                    out_data_8=8'd0;
                    if (location_bits < 3'd6)
                        data_bits_16 = (_in_data >> (31 - location_bits)) & ((1 << (location_bits)) - 1);
//                        data_bits_16 =  data_bits_16 >> (6 - location_bits);
                   else begin
                        data_bits_16[5:1] = _in_data[30 : 26];
                        data_bits_16[0] =  _in_data[25] | _in_data[24];
                    end




                    parity_bit = ^data_bits_16; // Even parity calculation

                    // Error correction bits calculation
                    error_correction_bits[3] = location_bits[4] ^ location_bits[2] ^ location_bits[1];
                    error_correction_bits[2] = location_bits[4] ^ location_bits[3] ^ location_bits[1];
                    error_correction_bits[1] = location_bits[4] ^ location_bits[3] ^ location_bits[2];
                    error_correction_bits[0] = location_bits[0];

                    // Combine all fields into the output
                    out_data_16 = {parity_bit, location_bits, error_correction_bits, data_bits_16};
                end


                // case_2:32 bit to 12 bit approximation
                b: begin
                    out_data_16=16'd0;
                    out_data_8=8'd0;

                    if (location_bits < 3'd3) begin

                        data_bits_12 = (_in_data >> (31 - location_bits)) & ((1 << (location_bits)) - 1);
                    end


                    else begin
                        data_bits_12[2:1] = _in_data[30 : 29];
                        data_bits_12[0] =  _in_data[28] | _in_data[27];

                    end

                    error_correction_bits[3] = location_bits[4] ^ location_bits[2] ^ location_bits[1];
                    error_correction_bits[2] = location_bits[4] ^ location_bits[3] ^ location_bits[1];
                    error_correction_bits[1] = location_bits[4] ^ location_bits[3] ^ location_bits[2];
                    error_correction_bits[0] = location_bits[0];

                    out_data_12 = {location_bits, error_correction_bits, data_bits_12};
                end


                // case_3:32 bit to 8 bit approximation
                c: begin
                    out_data_16=16'd0;
                    out_data_12=12'd0;

                    if (location_bits == 3'd1) begin
                        data_bits_8[1:0] = {1'b0, _in_data[30]};
                    end
                    else begin
                        data_bits_8[1] = _in_data[30];
                        data_bits_8[0] =  _in_data[29] | _in_data[28];
                    end



                    parity_bit = ^location_bits;

                    out_data_8 = {location_bits, parity_bit, data_bits_8};
                end
            endcase
        end
    end
endmodule




