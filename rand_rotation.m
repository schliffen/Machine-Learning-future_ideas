function r_data = rand_rotation(Data, rtheta)
   Data_x = Data(:,1:100);
   Data_y = Data(:,101:200);

   sinus   = sin(rtheta*pi/180);
   cosinus = cos(rtheta*pi/180);
   r_data =[Data_x*cosinus + Data_y*sinus - Data_x*sinus + Data_y *cosinus ...
            Data_x * cosinus + Data_y*sinus - Data_x*sinus + Data_y * cosinus];
               
end