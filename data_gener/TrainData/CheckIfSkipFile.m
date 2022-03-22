function [skipFile] = CheckIfSkipFile(sceneName, ii)
%CheckIfSkipFile 
%   Skip some files if they were giving seg fault in my configuration.
        skipFile = false;
        if(strcmp(sceneName, 'living_room_0059'))
            if((ii == 195) || (ii == 204) || (ii == 208))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'living_room_0057'))
            if((ii == 123))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'office_0001c'))
            if((ii == 83))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'dining_room_0010'))
            if((ii == 27))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'living_room_0040'))
            if((ii == 32) || (ii == 61) || (ii == 87))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'living_room_0073'))
            if((ii == 60))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'dining_room_0020'))
            if((ii == 120))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'dining_room_0022'))
            if((ii == 30))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'dining_room_0024'))
            if((ii == 25) || (ii == 29))
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'study_room_0003'))
            if((ii == 1) || (ii == 7) || (ii == 8) || (ii == 10) || (ii == 24) || (ii == 29)|| (ii == 61) || (ii == 85)|| (ii == 97)|| (ii == 101) )
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'study_room_0005b'))
            if((ii == 45) )
                skipFile = true;
            end
        elseif(strcmp(sceneName, 'study_room_0007'))
            if((ii == 17) )
                skipFile = true;
            end
        end
end