#include <ctime>
#include "celex5_ros.h"

namespace celex_ros {

CelexRos::CelexRos() {}

int CelexRos::initDevice() {
  init_time = ros::Time::now();
  frame_interval = ros::Duration(30/1000000);
  frame_cnt = 0;
  previous_frame_len = 0;
  previous_x = 0;
  previous_y = 0;
}

void CelexRos::set_frame_interval(uint32_t i) {
  frame_interval = ros::Duration(double(i)/1000000);
}

double CelexRos::get_frame_interval() {
  return frame_interval.toSec();
}

void CelexRos::grabEventData(
    CeleX5 *celex,
    celex5_msgs::eventData &msg,
    double max_time_diff,
    uint32_t *vectorIndex) {
  if (celex->getSensorFixedMode() == CeleX5::Event_Off_Pixel_Timestamp_Mode) {
    std::clock_t timeBegin;
    double timeDiff = 0.0;
    timeBegin = std::clock();

    *vectorIndex = 0;
    uint32_t vectorLength = 0;

    while (timeDiff < max_time_diff) {
      std::vector<EventData> vecEvent;
      celex->getEventDataVector(vecEvent);

      timeDiff = (std::clock() - timeBegin)/(double) CLOCKS_PER_SEC;
      
      uint32_t dataSize = vecEvent.size();
      if (dataSize == 0) {
        continue;
      }
      uint32_t first_x = vecEvent[0].col;
      uint32_t first_y = vecEvent[0].row;
      if (dataSize==previous_frame_len && first_x==previous_x && first_y==previous_y) {
        continue;
      }
      previous_frame_len = dataSize;
      previous_x = first_x;
      previous_y = first_y;

      vectorLength += dataSize;

      for (int i = 0; i < dataSize; i++) {
        msg.x.push_back(vecEvent[i].col);
        msg.y.push_back(MAT_ROWS - vecEvent[i].row - 1);
        // msg.timestamp = init_time + ros::Duration(double(vecEvent[i].t_off_pixel)*0.000014).toSec();
        msg.timestamp.push_back(0); 
      }
      init_time += frame_interval;
    }
    if (vectorLength == 0)
      return;
    frame_cnt += 1;
    *vectorIndex = frame_cnt;
  } else {
    std::cout << "This mode has no event data. " << std::endl;
  }
}

CelexRos::~CelexRos() {}
}
