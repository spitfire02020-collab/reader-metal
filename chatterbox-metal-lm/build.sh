#!/bin/bash
# ChatterboxMetalLM Build Notes
#
# On iOS, Metal files are compiled automatically by Xcode's build system.
# No manual xcrun metallib invocation needed.
#
# To integrate into Reader app:
# 1. Open Reader.xcodeproj in Xcode
# 2. Drag chatterbox-metal-lm/src/ into the Reader target
# 3. Or add to project.yml sources if using XcodeGen
#
# The Swift files in chatterbox-metal-lm/src/ should be added to the
# same target that includes LanguageModel.metal