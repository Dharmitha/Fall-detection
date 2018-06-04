clear all;


Input = readtable('SDLFeatures.csv');
X = Input(:,1:end);

X_t = table2array(X);

nC = size(X_t,2);
for i = 1:nC
    M = (max(X_t(:,i)) - min(X_t(:,i)));
    if M ~= 0
        X_t2(:,i) = (X_t(:,i) - min(X_t(:,i))) / M;
    else
        X_t2(:,i) = X_t(:,i);
    end
end

X_t3 = array2table(X_t2);
T2 = [X_t3];
T2.Properties.VariableNames = Input.Properties.VariableNames;

writetable(T2,'SDLFeatures_normalized.csv');
