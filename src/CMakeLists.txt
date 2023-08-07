add_library(demoam SHARED
            backend.cpp
            camera.cpp
            config.cpp
            dataset.cpp
            feature.cpp
            frame.cpp
            frontend.cpp
            imu_types.cpp
            map.cpp
            mappoint.cpp
            viewer.cpp
            visual_odometry.cpp
            )
target_link_libraries(demoam
            ${THIRD_PARTY_LIBS}
            )