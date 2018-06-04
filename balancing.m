clear all;

Input = readtable('FOL_FKL_BSC_SDL_STD_LYI.csv');
nR = size(Input,1);
X = Input(:,1:end-1);
Y = Input(:,end);
Y_t = table2array(Y);
for i = 1:size(Y_t,1)
    if strcmp(Y_t{i},'FOL') == 1 || strcmp(Y_t{i},'FKL') == 1 || strcmp(Y_t{i},'BSC') == 1 || strcmp(Y_t{i},'SDL') == 1 
        Y_t{i} = 'Fall';
    elseif strcmp(Y_t{i},'STD') == 1 || strcmp(Y_t{i},'LYI') == 1 
        Y_t{i} = 'Non-Fall';
    end
end

Y_t2 = array2table(Y_t);
Changed = [X Y_t2];
%writetable(Changed, 'Changed.csv');

index = find(strcmp(Y_t,'Fall'));
Fall = Changed(index,:);
NonFall = Changed(setdiff(1:nR,index),:);
Fall_b = datasample(Fall, 2000);
NonFall_b = datasample(NonFall,2000,'Replace',false);

%%
Balanced = vertcat(Fall_b,NonFall_b);
writetable(Balanced,'Balanced_STD_LYI.csv');



