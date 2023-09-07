BUILD_DIR = build
HEADERS = ./headers/
VPATH = src:headers

COMPILER = g++
C_FLAGS = -c -Wall
C_INCLUDES = -I$(HEADERS)

PROJECT = perceptron
SOURCES = main.cpp perceptron.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))

$(BUILD_DIR)/$(PROJECT) : $(OBJECTS)
	$(COMPILER) -o $@ $(OBJECTS)

$(BUILD_DIR)/%.o : %.cpp | $(BUILD_DIR)
	$(COMPILER) -g $(C_FLAGS) $(C_INCLUDES) -o $@ $<

$(BUILD_DIR) :
	mkdir $(BUILD_DIR)

runProject :
	./$(BUILD_DIR)/$(PROJECT)

debugProject:
	gdb ./$(BUILD_DIR)/$(PROJECT)

clean :
	rm $(BUILD_DIR)/* >/dev/null 2>&1