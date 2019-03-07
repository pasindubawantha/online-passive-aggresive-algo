data = load("datafile.csv");
x = data(:,[2:10]);
y = data(:,11);
y(y == 2) = -1;
y(y == 4) = 1;


training_ratio = 2/3;
training_size = round(size(x)(1)*training_ratio);
testing_size = size(x)(1) - training_size;

x_training = x([1:training_size],:);
x_testing = x([training_size:size(x)(1)],:);

y_training = y([1:training_size],:);
y_testing = y([training_size:size(y)(1)],:);


# Passive Aggresive 
function weights = passive_aggresive(x_training,y_training,epochs)
  C = 1;
  # epochs = 1; # 2 ,10
  dimentions = size(x_training)(2);

  w = zeros(dimentions)(1,:);

  for epoch = 1:epochs
    for instance = 1:size(x_training)(1)
      x_t = x_training(instance,:)';
      y_hat = w*x_t;
      y_t = y_training(instance,:);
      
      lost = 1 - y_t*y_hat;
      if lost < 0
        lost = 0;
      end
      
      x_length = 0;
      for d=1:size(x_t)(1)
      x_length += x_t(d)^2;
      end
      
      x_length =sqrt(x_length);
      T = lost/x_length;
      
      if C<T
        T = C;
      end
      
      x_t = x_t';
      w = w + T*y_t*x_t;
    end
  end
weights = w';
endfunction

weights = passive_aggresive(x_training,y_training,1)
for i=1:size(x_testing)(1)
y = x_testing(i,:)*weights
end