import smbus2 as smbus  # Istedigin gibi smbus2 kullaniyoruz
import time

# I2C address
I2C_ADDRESS = 0x57

# Register addresses
REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_PILOT_PA = 0x10
REG_MULTI_LED_CTRL1 = 0x11
REG_MULTI_LED_CTRL2 = 0x12
REG_PART_ID = 0xFF

class MAX30102:
    def __init__(self, channel=1, address=I2C_ADDRESS):
        self.address = address
        self.bus = smbus.SMBus(channel)
        
        # 1. Reset at ve bekle
        self.reset()
        time.sleep(1)

        # 2. Interruptlari ayarla
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_1, 0xc0)
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_2, 0x00)
        
        # 3. FIFO Konfigurasyonu
        # SMP_AVE=0 (Average yok), ROLLOVER_EN=1
        self.bus.write_byte_data(self.address, REG_FIFO_CONFIG, 0x1f)
        
        # 4. SpO2 Konfigurasyonu (Hiz Ayari)
        # SPO2_SR=100Hz (0x27)
        self.bus.write_byte_data(self.address, REG_SPO2_CONFIG, 0x27)
        
        # 5. LED Parlakliklari (DUZELTME BURADA)
        # 0x3F sensuru doyuma ulastirdi, 0x1F (yaklasik 6.4mA) daha guvenli.
        # Sinyal hala duz cizgi cikarsa bunu 0x10 a kadar dusebiliriz.
        self.bus.write_byte_data(self.address, REG_LED1_PA, 0x1F) # RED
        self.bus.write_byte_data(self.address, REG_LED2_PA, 0x1F) # IR
        self.bus.write_byte_data(self.address, REG_PILOT_PA, 0x7f)
        
        # 6. MODU AYARLA (SpO2 modu)
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x03)
        
        # 7. POINTERLARI SIFIRLA
        self.bus.write_byte_data(self.address, REG_FIFO_WR_PTR, 0x00)
        self.bus.write_byte_data(self.address, REG_OVF_COUNTER, 0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_RD_PTR, 0x00)

    def reset(self):
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x40)

    def available(self):
        """FIFO'da veri var mi kontrol eder."""
        try:
            read_ptr = self.bus.read_byte_data(self.address, REG_FIFO_RD_PTR)
            write_ptr = self.bus.read_byte_data(self.address, REG_FIFO_WR_PTR)
            
            if read_ptr == write_ptr:
                return 0
            
            num_samples = write_ptr - read_ptr
            if num_samples < 0:
                num_samples += 32
            
            return num_samples
        except:
            return 0

    def read_fifo(self):
        """Veri okur."""
        try:
            data = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)
            
            # Byte birlestirme
            red_led = (data[0] << 16 | data[1] << 8 | data[2]) & 0x03FFFF
            ir_led = (data[3] << 16 | data[4] << 8 | data[5]) & 0x03FFFF
            
            return red_led, ir_led
        except Exception:
            return -1, -1

    def shutdown(self):
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x80)
